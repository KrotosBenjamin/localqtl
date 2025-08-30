"""
GPU-accelerated association mapping core utilities.
Provides residualization, allele frequency utilities,
correlation/regression functions, beta approximation
for permutation p-values, and I/O helpers.
"""
import torch
import sys, re
import numpy as np
import pandas as pd
import scipy.optimize
from time import strftime
import scipy.stats as stats
from scipy.special import loggamma
from typing import Optional, Tuple
from collections import OrderedDict

from helper import _prepare_tensor
# ----------------------------
# Output dtype specification
# ----------------------------
output_dtype_dict = {
    'num_var':np.int32,
    'beta_shape1':np.float32,
    'beta_shape2':np.float32,
    'true_df':np.float32,
    'pval_true_df':np.float64,
    'variant_id':str,
    'start_distance':np.int32,
    'end_distance':np.int32,
    'ma_samples':np.int32,
    'ma_count':np.int32,
    'af':np.float32,
    'pval_nominal':np.float64,
    'slope':np.float32,
    'slope_se':np.float32,
    'pval_perm':np.float64,
    'pval_beta':np.float64,
}

# ----------------------------
# Simple logging
# ----------------------------
class SimpleLogger(object):
    """
    Simple logger that writes timestamped messages to console and optionally to a file.
    Supports context manager usage to ensure logfile is always closed.

    Example
    -------
    with SimpleLogger("analysis.log") as logger:
        logger.write("Starting cis-mapping")
    """

    def __init__(self, logfile: str = None, verbose: bool = True):
        self.console = sys.stdout
        self.verbose = verbose
        self.log = open(logfile, "w") if logfile else None

    def _timestamp(self) -> str:
        """Return current timestamp as string."""
        return strftime("%Y-%m-%d %H:%M:%S")

    def write(self, message: str):
        """Write a timestamped message to console and logfile (if provided)."""
        msg = f"[{self._timestamp()}] {message}"
        if self.verbose:
            self.console.write(msg + "\n")
        if self.log is not None:
            self.log.write(msg + "\n")
            self.log.flush()

    def close(self):
        """Close the logfile if open."""
        if self.log is not None:
            self.log.close()
            self.log = None

    # Context manager methods
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

# -----------------------------------------------------------------------------
# Residualization and normalization utilities
# -----------------------------------------------------------------------------
class Residualizer(object):
    """
    Residualizer for regressing out covariates from genotype/phenotype matrices.

    Parameters
    ----------
    C_t : torch.Tensor
        Covariates (samples x covariates).
    """
    def __init__(self, C_t: torch.Tensor):
        # Center and orthogonalize covariates
        self.Q_t, _ = torch.linalg.qr(C_t - C_t.mean(0))
        self.dof = C_t.shape[0] - 2 - C_t.shape[1]

    def transform(self, M_t: torch.Tensor, center: bool=True) -> torch.Tensor:
        """Residualize rows of M wrt columns of C"""
        M0_t = M_t - M_t.mean(1, keepdim=True)
        if center:
            return M0_t - torch.mm(torch.mm(M0_t, self.Q_t), self.Q_t.t())
        else:
            return M_t - torch.mm(torch.mm(M0_t, self.Q_t), self.Q_t.t())


def center_normalize(M_t: torch.Tensor, dim: int=0) -> torch.Tensor:
    """Center and normalize matrix (M) along given dimension"""
    N_t = M_t - M_t.mean(dim=dim, keepdim=True)
    return N_t / torch.sqrt(torch.pow(N_t, 2).sum(dim=dim, keepdim=True))


# -----------------------------------------------------------------------------
# Allele frequency and filtering utilities
# -----------------------------------------------------------------------------
def calculate_maf(genotype_t: torch.Tensor, alleles: int=2):
    """Calculate minor allele frequency"""
    af_t = genotype_t.sum(1) / (alleles * genotype_t.shape[1])
    return torch.where(af_t > 0.5, 1 - af_t, af_t)


def get_allele_stats(genotype_t: torch.Tensor):
    """Compute allele frequency, minor allele samples, and minor allele counts."""
    # allele frequency
    n2 = 2 * genotype_t.shape[1]
    af_t = genotype_t.sum(1) / n2
    ix_t = af_t <= 0.5
    m = genotype_t > 0.5
    a = m.sum(1).int()
    b = (genotype_t < 1.5).sum(1).int()
    ma_samples_t = torch.where(ix_t, a, b)
    a = (genotype_t * m.float()).sum(1).int()
    ma_count_t = torch.where(ix_t, a, n2-a)
    return af_t, ma_samples_t, ma_count_t


def filter_maf(genotypes_t, variant_ids, maf_threshold, alleles=2):
    """Filter variants failing MAF threshold."""
    af_t = genotypes_t.sum(1) / (alleles * genotypes_t.shape[1])
    maf_t = torch.where(af_t > 0.5, 1 - af_t, af_t)
    if maf_threshold > 0:
        mask_t = maf_t >= maf_threshold
        genotypes_t = genotypes_t[mask_t]
        variant_ids = variant_ids[mask_t.cpu().numpy().astype(bool)]
        af_t = af_t[mask_t]
    return genotypes_t, variant_ids, af_t


def filter_maf_interaction(
        genotypes_t: torch.Tensor,
        interaction_mask_t: Optional[torch.Tensor] = None,
        maf_threshold_interaction: float = 0.05):
    """Filter genotypes for interaction tests to avoid colinearity."""
    mask_t = ~((genotypes_t == 0).all(1) | (genotypes_t == 1).all(1) | (genotypes_t == 2).all(1))

    if interaction_mask_t is not None:
        upper_t = calculate_maf(genotypes_t[:, interaction_mask_t]) >= maf_threshold_interaction - 1e-7
        lower_t = calculate_maf(genotypes_t[:,~interaction_mask_t]) >= maf_threshold_interaction - 1e-7
        mask_t = mask_t & upper_t & lower_t

    return genotypes_t[mask_t], mask_t


def impute_mean(genotypes_t, missing=-9):
    """Impute missing genotypes to row means."""
    m = genotypes_t == missing
    ix = torch.nonzero(m, as_tuple=True)[0]
    if len(ix) > 0:
        a = genotypes_t.sum(1)
        b = m.sum(1).float()
        mu = (a - missing*b) / (genotypes_t.shape[1] - b)
        genotypes_t[m] = mu[ix]

# -----------------------------------------------------------------------------
# Association testing utilities
# -----------------------------------------------------------------------------
def calculate_corr(genotype_t, phenotype_t, residualizer=None, return_var=False):
    """Calculate correlation between normalized residual genotypes and phenotypes"""
    if residualizer is not None:
        genotype_res_t = residualizer.transform(genotype_t)  # variants x samples
        phenotype_res_t = residualizer.transform(phenotype_t)  # phenotypes x samples
    else:
        genotype_res_t = genotype_t
        phenotype_res_t = phenotype_t

    if return_var:
        genotype_var_t = genotype_res_t.var(1)
        phenotype_var_t = phenotype_res_t.var(1)

    # Center and normalize
    genotype_res_t = center_normalize(genotype_res_t, dim=1)
    phenotype_res_t = center_normalize(phenotype_res_t, dim=1)

    # Correlation
    cor_t = torch.mm(genotype_res_t, phenotype_res_t.t())
    if return_var:
        return cor_t, genotype_var_t, phenotype_var_t
    else:
        return cor_t


def get_t_pval(t, df, log=False):
    """
    Get two-sided p-value from t-statistic.

    If log=True, returns -log10(P) while staying in log-space for stability at
    very small p-values.
    """
    if not log:
        return 2 * stats.t.cdf(-abs(t), df)
    else:
        log_pval = stats.t.logcdf(-np.abs(t), df) + np.log(2.0)
        return -log_pval / np.log(10.0)

# -----------------------------------------------------------------------------
# Regression, filtering, and covariates
# -----------------------------------------------------------------------------
def calculate_association(genotype_df, phenotype_s, covariates_df=None,
                          interaction_s=None, haplotype_df=None,
                          maf_threshold_interaction=0.05,
                          logp=False, window=1_000_000, verbose=True):
    """
    Standalone helper function for computing the association between
    genotypes/haplotypes and a phenotype.

    This does not work yet.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert genotype_df.columns.equals(phenotype_s.index)

    # GPU tensors
    phenotype_t = _prepare_tensor(phenotype_s.values, device=device)
    genotypes_t = _prepare_tensor(genotype_df.values, device=device)
    impute_mean(genotypes_t)

    haplotypes_t = None
    if haplotype_df is not None:
        assert haplotype_df.columns.equals(phenotype_s.index)
        haplotypes_t = _prepare_tensor(haplotype_df.values, device=device)

    dof = phenotype_s.shape[0] - 2
    residualizer = None
    if covariates_df is not None:
        assert phenotype_s.index.equals(covariates_df.index)
        cov_t = _prepare_tensor(covariates_df.values, device=device)
        residualizer = Residualizer(cov_t)
        dof -= covariates_df.shape[1]

    # Simple cis model
    if interaction_s is None:
        res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer,
                                    haplotypes_t=haplotypes_t)
        if len(res) == 6:
            tstat, slope, slope_se, af, ma_samples, ma_count = [i.cpu().numpy() for i in res]
            df = pd.DataFrame({
                'pval_nominal': get_t_pval(tstat, dof, log=logp),
                'slope': slope, 'slope_se': slope_se,
                'tstat': tstat, 'af': af, 'ma_samples': ma_samples, 'ma_count': ma_count,
            }, index=genotype_df.index)
        else:
            tstat, slope, slope_se = [i.cpu().numpy() for i in res]
            df = pd.DataFrame({
                'pval_nominal': get_t_pval(tstat, dof, log=logp),
                'slope': slope, 'slope_se': slope_se,
                'tstat': tstat,
            }, index=genotype_df.index)
    else:
        interaction_t = _prepare_tensor(interaction_s.values.reshape(1, -1),
                                        device=device)
        if maf_threshold_interaction > 0:
            mask_s = pd.Series(True, index=interaction_s.index)
            mask_s[interaction_s.sort_values(kind='mergesort').index[:interaction_s.shape[0]//2]] = False
            interaction_mask_t = torch.BoolTensor(mask_s.values).to(device)
        else:
            interaction_mask_t = None

        genotypes_t, mask_t = filter_maf_interaction(genotypes_t, interaction_mask_t=interaction_mask_t,
                                                     maf_threshold_interaction=maf_threshold_interaction)
        res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0),
                                            interaction_t, residualizer,
                                            haplotypes_t=haplotypes_t,
                                            return_sparse=False)
        tstat, b, b_se, af, ma_samples, ma_count = [i.cpu().numpy() for i in res]
        mask = mask_t.cpu().numpy()
        dof -= 2
        df = pd.DataFrame({
            'pval_g': get_t_pval(tstat[:,0], dof, log=logp),
            'b_g': b[:,0], 'b_g_se': b_se[:,0],
            'pval_i': get_t_pval(tstat[:,1], dof, log=logp),
            'b_i': b[:,1], 'b_i_se': b_se[:,1],
            'pval_gi': get_t_pval(tstat[:,2], dof, log=logp),
            'b_gi': b[:,2], 'b_gi_se': b_se[:,2],
            'af':af, 'ma_samples':ma_samples, 'ma_count':ma_count,
        }, index=genotype_df.index[mask])

    if df.index.str.startswith('chr').all():  # assume chr_pos_ref_alt_build format
        df['position'] = df.index.map(lambda x: int(x.split('_')[1]))

    return df


def prepare_cis_output(r_nominal, r2_perm, std_ratio, g, num_var, dof, variant_id,
                       start_distance, end_distance, phenotype_id, nperm=10_000,
                       logp=False):
    """Return nominal p-value, allele frequencies, etc. as pd.Series"""
    r2_nominal = r_nominal * r_nominal
    pval_perm = (np.sum(r2_perm >= r2_nominal) + 1) / (nperm + 1)

    slope = r_nominal * std_ratio
    tstat2 = dof * r2_nominal / (1 - r2_nominal)
    slope_se = np.abs(slope) / np.sqrt(tstat2)

    n2 = 2 * len(g)
    af = np.sum(g) / n2
    if af <= 0.5:
        ma_samples = np.sum(g>0.5)
        ma_count = np.sum(g[g>0.5])
    else:
        ma_samples = np.sum(g<1.5)
        ma_count = n2 - np.sum(g[g>0.5])

    res_s = pd.Series(OrderedDict([
        ('num_var', num_var),
        ('beta_shape1', np.nan),
        ('beta_shape2', np.nan),
        ('true_df', np.nan),
        ('pval_true_df', np.nan),
        ('variant_id', variant_id),
        ('start_distance', start_distance),
        ('end_distance', end_distance),
        ('ma_samples', ma_samples),
        ('ma_count', ma_count),
        ('af', af),
        ('pval_nominal', pval_from_corr(r2_nominal, dof, logp=logp)),
        ('slope', slope),
        ('slope_se', slope_se),
        ('pval_perm', pval_perm),
        ('pval_beta', np.nan),
    ]), name=phenotype_id)
    return res_s


def calculate_cis_nominal(genotypes_t, phenotype_t, residualizer=None,
                          haplotypes_t=None, return_af=True):
    """
    Compute nominal cis-association statistics.

    genotypes_t: genotypes x samples
    phenotype_t: single phenotype
    residualizer: Residualizer object (see core.py)
    haplotypes_t: haplotypes x samples
    """
    # Concatenate haplotypes if provided
    if haplotypes_t is not None:
        X_t = torch.cat([genotypes_t, haplotypes_t], dim=0)  # (variants+haplotypes) x samples
    else:
        X_t = genotypes_t

    # Ensure phenotypes is 2D
    p = phenotype_t.view(1, -1)

    # Correlation and variance components
    r_nominal_t, genotype_var_t, phenotype_var_t = calculate_corr(
        X_t, p, residualizer=residualizer, return_var=True
    )
    r_nominal_t = r_nominal_t.squeeze()
    r2_nominal_t = r_nominal_t.double().pow(2)

    # Degrees of freedom
    dof = residualizer.dof if residualizer is not None else p.shape[1] - 2

    # Effect sizes
    std_ratio_t = torch.sqrt(phenotype_var_t / genotype_var_t).squeeze()
    slope_t = r_nominal_t * std_ratio_t
    tstat_t = r_nominal_t * torch.sqrt(dof / (1 - r2_nominal_t))
    slope_se_t = (slope_t.double() / tstat_t).float()

    if not return_af:
        return tstat_t, slope_t, slope_se_t

    # Allele frequency stats for variants only
    af_t, ma_samples_t, ma_count_t = get_allele_stats(genotypes_t)
    return tstat_t, slope_t, slope_se_t, af_t, ma_samples_t, ma_count_t


def calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                               residualizer=None, haplotypes_t=None,
                               random_tiebreak=False, eps=1e-12):
    """
    Compute nominal correlation and empirical permutation-based correlations.
    """
    # Combine genotypes + haplotypes if both provided
    if haplotypes_t is not None:
        X_t = torch.cat([genotypes_t, haplotypes_t], dim=0)
    else:
        X_t = genotypes_t

    # Generate permuted phenotypes
    permutations_t = phenotype_t[permutation_ix_t]

    # Nominal correlation
    r_nominal_t, genotype_var_t, phenotype_var_t =  calculate_corr(
        X_t, phenotype_t.view(1,-1), residualizer=residualizer, return_var=True
    )
    r_nominal_t = r_nominal_t.squeeze()
    std_ratio_t = torch.sqrt(phenotype_var_t / genotype_var_t.clamp(min=eps)).squeeze()

    # Permutation correlations
    corr_t = calculate_corr(X_t, permutations_t, residualizer=residualizer).pow(2)
    corr_t = corr_t[~torch.isnan(corr_t).any(1)]
    if corr_t.shape[0] == 0:
        raise ValueError('All correlations resulted in NaN. Please check phenotype values.')
    r2_perm_t, _ = corr_t.max(0)

    # Nominal r^2
    r2_nominal_t = r_nominal_t.pow(2)
    r2_nominal_t[torch.isnan(r2_nominal_t)] = -1  # Workaround for nanargmax()

    # Select best row (variant or haplotype)
    if random_tiebreak:
        max_ix = torch.nonzero(r2_nominal_t == r2_nominal_t.max(), as_tuple=True)[0]
        ix = max_ix[torch.randint(0, len(max_ix), (1,))[0]]
    else:
        ix = r2_nominal_t.argmax()

    return r_nominal_t[ix], std_ratio_t[ix], ix, r2_perm_t, X_t[ix]


def calculate_interaction_nominal(
        genotypes_t: torch.Tensor,
        phenotypes_t: torch.Tensor,
        interaction_t: torch.Tensor,
        haplotypes_t: Optional[torch.Tensor] = None,
        residualizer: Optional[object] = None,
        return_sparse: bool = False,
        tstat_threshold: Optional[float] = None,
        variant_ids: Optional[np.ndarray] = None
) -> Tuple:
    """
    Extended interaction model:
        Y ~ G + H + I + (G×I) + (H×I)
    where:
        - G: genotype dosage
        - H: haplotype/ancestry dosage
        - I: interaction covariate(s)

    Parameters
    ----------
    genotypes_t : [num_genotypes x num_samples] tensor
    phenotypes_t : [num_phenotypes x num_samples] tensor
    interaction_t : [num_samples x num_interactions] tensor
    haplotypes_t : optional, [num_haplotypes x num_samples] tensor
    residualizer : optional, Residualizer
    return_sparse : if True, return only significant interactions
    tstat_threshold : threshold for sparse return
    variant_ids : optional, list of variant IDs for debugging

    Returns
    -------
    If return_sparse=False:
        tstat_t, b_t, b_se_t
    Else:
        (sparse outputs for interaction terms only)
    """
    n_var, n_samp = genotypes_t.shape
    n_hap = haplotypes_t.shape[0]
    n_pheno = phenotypes_t.shape[0]
    n_int = interaction_t.shape[1]

    # Center all inputs
    g0_t = genotypes_t - genotypes_t.mean(1, keepdim=True)      # variants x samples
    h0_t = haplotypes_t - haplotypes_t.mean(1, keepdim=True)    # haplotypes x samples
    i0_t = interaction_t - interaction_t.mean(0, keepdim=True)  # samples x interactions
    p0_t = phenotypes_t - phenotypes_t.mean(1, keepdim=True)    # phenotypes x samples

    # Construct interaction term and center
    gi_t = g0_t.unsqueeze(2) * i0_t.unsqueeze(0)   # (n_var x n_samp x n_int)
    gi0_t = gi_t - gi_t.mean(1, keepdim=True)      # center across samples

    hi_t = h0_t.unsqueeze(2) * i0_t.unsqueeze(0)   # (n_hap x n_samp x n_int)
    hi0_t = hi_t - hi_t.mean(1, keepdim=True)

    # Residualize if covariates exist
    if residualizer is not None:
        p0_t = residualizer.transform(p0_t, center=False)
        g0_t = residualizer.transform(g0_t, center=False)
        h0_t = residualizer.transform(h0_t, center=False)
        i0_t = residualizer.transform(i0_t.t(), center=False).t()
        for k in range(n_int):
            gi0_t[..., k] = residualizer.transform(gi0_t[..., k], center=False)
            hi0_t[..., k] = residualizer.transform(hi0_t[..., k], center=False)

    # Build design matrix -- block struction
    X_parts = [
        g0_t.unsqueeze(-1),            # (n_var x n_samp x 1)
        h0_t.unsqueeze(0).repeat(n_var, 1, 1).transpose(1, 2),  # broadcast haplotypes
        i0_t.repeat(n_var, 1, 1),      # (n_var x n_samp x n_int)
        gi0_t,                         # (n_var x n_samp x n_int)
        hi0_t.repeat(n_var, 1, 1)      # expand hap interactions per variant
    ]
    X_t = torch.cat(X_parts, dim=2)   # shape: (n_var x n_samp x nb)

    # Regression
    try:
        XtX = torch.matmul(X_t.transpose(1, 2), X_t) # (n_var x nb x nb)
        Xinv = torch.linalg.inv(XtX)
    except RuntimeError as e:
        if variant_ids is not None and len(e.args) >= 1:
            # annotate failure with variant ID
            m = re.search(r"For batch (\d+)", str(e))
            if m:
                idx = int(m.group(1))
                e.args = (e.args[0] + f'\n    Likely problematic variant: {variant_ids[idx]} ',) + e.args[1:]
        raise

    p_tile = p0_t.unsqueeze(0).expand([n_var, *p0_t.shape])  # (n_var x n_pheno x n_samp)

    # Fit beta coefficients
    b_t = torch.matmul(torch.matmul(XtX_inv, X_t.transpose(1, 2)), p_tile.transpose(1, 2))
    nb = b_t.shape[1]

    # Degrees of freedom
    if residualizer is not None:
        dof = residualizer.dof - (n_hap + 2*n_int + n_hap*n_int)
    else:
        dof = phenotypes_t.shape[1] - 2 - (n_hap + 2*n_int + n_hap*n_int)

    # Residuals and s.e.
    r_t = torch.matmul(X_t, b_t) - p_tile.transpose(1, 2)   # (n_var x n_samp x n_pheno)
    rss_t = (r_t * r_t).sum(1)                              # (n_var x n_pheno)
    diag_mask = torch.eye(nb, dtype=torch.bool, device=X_t.device)

    if nps == 1:  # single phenotype case
        b_t = b_t.squeeze(2)
        b_se_t = torch.sqrt(Xinv[:, diag_mask] * rss_t.unsqueeze(1) / dof)
    else: # multiple phenotypes
        b_se_t = torch.sqrt(
            Xinv[:, diag_mask].unsqueeze(-1).repeat([1,1,nps]) *
            rss_t.unsqueeze(1).repeat(1, nb, 1) / dof
        )

    # t-stats
    tstat_t = (b_t.double() / b_se_t.double()).float()

    # Output
    if not return_sparse:
        af_t, ma_samples_t, ma_count_t = get_allele_stats(genotypes_t)
        return tstat_t, b_t, b_se_t, af_t, ma_samples_t, ma_count_t
    else:  # sparse output
        raise NotImplementedError("Sparse mode not yet supported for haplotype data")


def linreg(X_t, y_t, dtype=torch.float64):
    """
    Robust linear regression with standardized X (first col = intercept).
    """
    # Standardize X
    x_std_t = X_t.std(0)
    x_mean_t = X_t.mean(0)
    x_std_t[0], x_mean_t[0] = 1, 0
    Xtilde_t = (X_t - x_mean_t) / x_std_t

    # Regression
    XtX_t = torch.matmul(Xtilde_t.T, Xtilde_t)
    Xty_t = torch.matmul(Xtilde_t.T, y_t)
    b_t = torch.linalg.solve(XtX_t, Xty_t.unsqueeze(-1)).squeeze()

    # Compute s.e.
    dof = X_t.shape[0] - X_t.shape[1]
    r_t = y_t - torch.matmul(Xtilde_t, b_t)
    sigma2_t = (r_t*r_t).sum() / dof
    XtX_inv_t = torch.linalg.solve(XtX_t, torch.eye(X_t.shape[1], dtype=dtype).to(X_t.device))
    var_b_t = sigma2_t * XtX_inv_t
    b_se_t = torch.sqrt(torch.diag(var_b_t))

    # Rescale and adjust intercept
    b_t /= x_std_t
    b_se_t /= x_std_t
    b_t[0] -= torch.sum(x_mean_t * b_t)
    ms_t = x_mean_t / x_std_t
    b_se_t[0] = torch.sqrt(b_se_t[0]**2 + torch.matmul(torch.matmul(ms_t.T, var_b_t), ms_t))
    return b_t, b_se_t


def filter_covariates(covariates_t: torch.Tensor, log_counts_t: torch.Tensor,
                      tstat_threshold: int = 2):
    """
    Filter covariates significantly associated with phenotype (remove intercept).
    """
    assert (covariates_t[:,0] == 0).all()
    b_t, b_se_t = linreg(covariates_t, log_counts_t)
    tstat_t = b_t / b_se_t
    m = tstat_t.abs() > tstat_threshold
    m[0] = False
    return covariates_t[:, m]


#------------------------------------------------------------------------------
#  Beta approximation functions
#------------------------------------------------------------------------------
def pval_from_corr(r2, dof, logp=False, eps=1e-12):
    """Convert squared correlation (r^2) to a two-sided p-value."""
    tstat2 = dof * r2 / np.clip(1 - r2, eps, None)
    return get_t_pval(np.sqrt(np.clip(tstat2, 0, None)), dof, log=logp)


def beta_shape_1_from_dof(r2, dof):
    """
    Estimate the first Beta distribution shape parameter via method-of-moments.
    """
    pval = pval_from_corr(r2, dof)
    mean = np.mean(pval, dtype=np.float64)
    var = np.var(pval, dtype=np.float64)
    if var <= 0:
        print("WARNING: Variance of p-values is zero; returning NaN for Beta.")
        return np.nan
    return mean * (mean * (1.0 - mean) / var - 1.0)


def beta_log_likelihood(x, shape1, shape2):
    """Negative log-likelihood of Beta distribution."""
    if len(x) == 0 or np.any((x <= 0) | (x >= 1)):
        print("WARNING: Input data to beta_log_likelihood is invalid; returning NaN.")
        return np.nan
    logbeta = loggamma(shape1) + loggamma(shape2) - loggamma(shape1 + shape2)
    ll = (shape1 - 1.0) * np.sum(np.log(x)) \
        + (shape2 - 1.0) * np.sum(np.log1p(-x)) \
        - len(x) * logbeta
    return -ll


def fit_beta_parameters(r2_perm, dof_init, tol=1e-4, return_minp=False):
    """
    Estimate beta parameters approximating permutation p-values.
    """
    try:
        log_true_dof = scipy.optimize.newton(
            lambda x: np.log(beta_shape_1_from_dof(r2_perm, np.exp(x))),
            np.log(dof_init), tol=tol, maxiter=50
        )
        true_dof = np.exp(log_true_dof)
    except Exception:
        print('WARNING: Newton root finding failed, falling back to Nelder-Mead')
        res = scipy.optimize.minimize(
            lambda x: np.abs(beta_shape_1_from_dof(r2_perm, x) - 1),
            dof_init, method='Nelder-Mead', tol=tol
        )
        true_dof = res.x[0]

    pval = pval_from_corr(r2_perm, true_dof)
    mean, var = np.mean(pval), np.var(pval)
    beta_shape1 = mean * (mean * (1 - mean) / var - 1)
    beta_shape2 = beta_shape1 * (1/mean - 1)
    res = scipy.optimize.minimize(
        lambda s: beta_log_likelihood(pval, s[0], s[1]),
        [beta_shape1, beta_shape2], method='Nelder-Mead', tol=tol
    )
    beta_shape1, beta_shape2 = res.x
    if return_minp:
        return beta_shape1, beta_shape2, true_dof, pval
    return beta_shape1, beta_shape2, true_dof


def calculate_beta_approx_pval(r2_perm, r2_nominal, dof_init, tol=1e-4):
    """
    Compute beta-approx empirical p-value for nominal r2.
    """
    beta_shape1, beta_shape2, true_dof = fit_beta_parameters(r2_perm, dof_init, tol)
    pval_true_dof = pval_from_corr(r2_nominal, true_dof)
    pval_beta = stats.beta.cdf(pval_true_dof, beta_shape1, beta_shape2)
    return pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof

#------------------------------------------------------------------------------
#  I/O functions
#------------------------------------------------------------------------------
def read_phenotype_bed(phenotype_bed):
    """Load phenotype BED file as phenotype and position DataFrames"""
    if phenotype_bed.lower().endswith(('.bed.gz', '.bed')):
        phenotype_df = pd.read_csv(phenotype_bed, sep='\t', index_col=3,
                                   dtype={'#chr':str, '#Chr':str})
    elif phenotype_bed.lower().endswith('.bed.parquet'):
        phenotype_df = pd.read_parquet(phenotype_bed)
        phenotype_df.set_index(phenotype_df.columns[3], inplace=True)
    else:
        raise ValueError('Unsupported file type.')

    phenotype_df.rename(columns={i:i.lower().replace('#chr','chr')
                                 for i in phenotype_df.columns[:3]},
                        inplace=True)
    phenotype_df['start'] += 1  # change to 1-based
    pos_df = phenotype_df[['chr', 'start', 'end']]
    phenotype_df.drop(['chr', 'start', 'end'], axis=1, inplace=True)

    # Validate BED sorting
    assert pos_df.equals(
        pos_df.groupby('chr', sort=False, group_keys=False)[pos_df.columns].apply(lambda x: x.sort_values(['start', 'end']))
    ), "Positions in BED file must be sorted."

    if (pos_df['start'] == pos_df['end']).all():
        pos_df = pos_df[['chr', 'end']].rename(columns={'end': 'pos'})

    return phenotype_df, pos_df
