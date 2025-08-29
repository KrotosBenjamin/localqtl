import torch
import numpy as np

from core import (
    Residualizer,
    prepare_cis_output,
    calculate_cis_permutations,
    calculate_beta_approx_pval
)
from utils import (
    _prepare_tensor,
    _prepare_window,
    _apply_maf_filters,
    _filter_monomorphic,
    _prepare_window_tensors
)

def _make_permutation_index(n_samples, nperm, seed, device, logger):
    ix = np.arange(n_samples)
    if seed is not None:
        np.random.seed(seed)
        logger.write(f"  * using seed {seed}")
    perm_ix = np.array([np.random.permutation(ix) for _ in range(nperm)])
    return torch.LongTensor(perm_ix).to(device)


def _process_group_permutations(buf, variant_df, start_pos, end_pos, dof,
                                group_id, nperm=10_000, beta_approx=True,
                                logp=False):
    """
    Merge results for grouped phenotypes

    buf: [r_nominal, std_ratio, var_ix, r2_perm, g, num_var, phenotype_id]
    """
    # select phenotype with strongest nominal association
    max_ix = np.argmax(np.abs([b[0] for b in buf]))
    r_nominal, std_ratio, var_ix = buf[max_ix][:3]
    g, num_var, phenotype_id = buf[max_ix][4:]

    # select best phenotype correlation for each permutation
    r2_perm = np.max([b[3] for b in buf], 0)

    # return r_nominal, std_ratio, var_ix, r2_perm, g, num_var, phenotype_id
    variant_id = variant_df.index[var_ix]
    start_distance = variant_df['pos'].values[var_ix] - start_pos
    end_distance = variant_df['pos'].values[var_ix] - end_pos
    res_s = prepare_cis_output(
        r_nominal, r2_perm, std_ratio, g, num_var, dof, variant_id,
        start_distance, end_distance, phenotype_id, nperm=nperm, logp=logp
    )
    if beta_approx:
        res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = \
            calculate_beta_approx_pval(r2_perm, r_nominal * r_nominal, dof * 0.25)
    res_s['group_id'] = group_id
    res_s['group_size'] = len(buf)
    return res_s


def _process_permutation_window(
        row, genotype_ix_t, variant_df, igc, mapping_state, permutation_ix_t,
        dof, covariates_df, paired_covariate_df, maf_threshold,
        beta_approx, random_tiebreak, logp, warn_monomorphic):
    """
    Handle cis-QTL mapping with permutations for a single phenotype window.
    Returns a DataFrame row of results or None if no valid variants.
    """
    # Unpack row
    device, residualizer, logger = (
        mapping_state["device"],
        mapping_state["residualizer"],
        mapping_state["logger"],
    )
    filt, g_idx, phenotype, phenotype_id = _prepare_window(
        row, genotype_ix_t, variant_df, igc, maf_threshold,
        device=device, logger=logger, warn_monomorphic=warn_monomorphic,
        is_group=False, is_perm=True
    )
    if filt is None:
        return None
    genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist = filt

    if genotypes_t.shape[0] == 0:
        return None

    # Residualizer (with paired covariates if present)
    if paired_covariate_df is None or phenotype_id not in paired_covariate_df:
        iresidualizer, idof = residualizer, dof
    else:
        pcov_t = _prepare_tensor(
            np.c_[covariates_df, paired_covariate_df[phenotype_id]], device=device
        )
        iresidualizer, idof = Residualizer(pcov_t), dof - 1

    # Run permutations
    phenotype_t = _prepare_tensor(phenotype, device=device)
    res = calculate_cis_permutations(
        genotypes_t, phenotype_t, permutation_ix_t,
        residualizer=iresidualizer, haplotypes_t=haplotypes_t,
        random_tiebreak=random_tiebreak
    )
    r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]

    # Build output
    var_ix = g_idx[var_ix]
    variant_id = variant_df.index[var_ix]
    start_distance = variant_df["pos"].values[var_ix] - igc.phenotype_start[phenotype_id]
    end_distance = variant_df["pos"].values[var_ix] - igc.phenotype_end[phenotype_id]
    res_s = prepare_cis_output(
        r_nominal, r2_perm, std_ratio, g,
        genotypes_t.shape[0], idof, variant_id,
        start_distance, end_distance, phenotype_id,
        nperm=len(permutation_ix_t), logp=logp
    )

    # Beta approximation (optional)
    if beta_approx:
        res_s[["pval_beta", "beta_shape1", "beta_shape2", "true_df", "pval_true_df"]] = \
            calculate_beta_approx_pval(r2_perm, r_nominal * r_nominal, idof)

    return res_s


def _process_group_permutation_window(
        row, genotype_ix_t, variant_df, igc, mapping_state, permutation_ix_t,
        dof, covariates_df, paired_covariate_df, maf_threshold, beta_approx,
        random_tiebreak, logp, warn_monomorphic
):
    """
    Handle cis-QTL mapping with permutations for grouped phenotypes.
    Returns a DataFrame row of results or None if no valid variants.
    """
    # Unpack row
    device, residualizer, logger = (
        mapping_state["device"],
        mapping_state["residualizer"],
        mapping_state["logger"],
    )
    filt, g_idx, phenotypes, phenotype_ids, group_id = _prepare_window(
        row, genotype_ix_t, variant_df, igc, maf_threshold,
        device=device, logger=logger, warn_monomorphic=warn_monomorphic,
        is_group=True, is_perm=True
    )
    if filt is None:
        return None
    genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist = filt

    if genotypes_t.shape[0] == 0:
        return None

    # Iterate over phenotypes
    buf = []
    for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
        phenotype_t = _prepare_tensor(phenotype, device=device)

        # Residualizer (with paired covariates if present)
        if paired_covariate_df is None or phenotype_id not in paired_covariate_df:
            iresidualizer, idof = residualizer, dof
        else:
            pcov_t = _prepare_tensor(
                np.c_[covariates_df, paired_covariate_df[phenotype_id]],
                device=device
            )
            iresidualizer, idof = Residualizer(pcov_t), dof - 1

        # Run permutations
        res = calculate_cis_permutations(
            genotypes_t, phenotype_t, permutation_ix_t,
            residualizer=iresidualizer, haplotypes_t=haplotypes_t,
            random_tiebreak=random_tiebreak
        )
        r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
        var_ix = g_idx[var_ix]
        buf.append([r_nominal, std_ratio, var_ix, r2_perm, g,
                    genotypes_t.shape[0], phenotype_id])

    # Process group-level summary
    return _process_group_permutations(
        buf, variant_df, igc.phenotype_start[phenotype_ids[0]],
        igc.phenotype_end[phenotype_ids[0]], dof,
        group_id, nperm=len(permutation_ix_t),
        beta_approx=beta_approx, logp=logp
    )
