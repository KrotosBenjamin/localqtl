import os
import time
import torch
import numpy as np
from collections import OrderedDict

import eigenmt
from core import (
    get_t_pval,
    impute_mean,
    Residualizer,
    SimpleLogger,
    calculate_maf,
    filter_maf_interaction
)

def _prepare_tensor(df, dtype=torch.float32, device='cpu'):
    return torch.tensor(df, dtype=dtype).to(device)


def _prepare_window_tensors(genotypes, haplotypes, genotype_ix_t, device):
    """Prepare genotype/haplotype tensors and impute missing."""
    genotypes_t = _prepare_tensor(genotypes, device=device)[:, genotype_ix_t]
    impute_mean(genotypes_t)

    haplotypes_t = None
    if haplotypes is not None:
        haplotypes_arr = _prepare_tensor(haplotypes, device=device)

        if haplotypes_arr.dim() == 2:
            haplotypes_t = haplotypes_arr[:, genotype_ix_t]

        elif haplotypes_arr.dim() == 3:
            n_var, n_samp, n_anc = haplotypes_arr.shape
            haplotypes_arr = haplotypes_arr[:, genotype_ix_t, :]

            if n_anc == 2:
                haplotypes_t = haplotypes_arr[:, :, 0]
            else:
                haplotypes_t = haplotypes_arr.reshape(n_variants, -1)
        else:
            raise ValueError(f"Unexpected haplotype tensor shape {haplotypes_arr.shape}")

    # Double-check variant alignment
    if haplotypes_t.shape[0] != genotypes_t.shape[0]:
        haplotypes_t = None

    return genotypes_t, haplotypes_t


def _log_mapping_context(genotype_df, haplotype_df, group_s, window,
                         random_tiebreak, logger):
    if group_s is not None:
        logger.write(f"  * {len(group_s.unique())} phenotype groups")
    logger.write(f"  * {genotype_df.shape[0]} variants")
    if haplotype_df is not None:
        logger.write("  * including haplotype/ancestry tracks")
    if random_tiebreak:
        logger.write("  * randomly selecting top variant in case of ties")
    logger.write(f"  * cis-window: ±{window:,}")


def _filter_monomorphic(genotypes_t, haplotypes_t, g_idx, warn, logger):
    mono_t = (genotypes_t == genotypes_t[:, [0]]).all(1)
    if mono_t.any():
        genotypes_t = genotypes_t[~mono_t]
        haplotypes_t = haplotypes_t[~mono_t] if haplotypes_t is not None else None
        g_idx = g_idx[~mono_t.cpu()]
        if warn:
            logger.write(f"    * WARNING: excluding {mono_t.sum()} monomorphic variants")
    return genotypes_t, haplotypes_t, g_idx


def _apply_maf_filters(genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist,
                       maf_threshold, interaction_df, maf_threshold_interaction):
    """Apply standard and interaction MAF filters, returning filtered tensors and arrays."""
    # Standard MAF filter
    if maf_threshold > 0:
        maf_t = calculate_maf(genotypes_t)
        mask_t = maf_t >= maf_threshold
        if mask_t.sum() == 0:
            return None
        genotypes_t = genotypes_t[mask_t]
        variant_ids = variant_ids[mask_t.cpu().numpy()]
        start_dist = start_dist[mask_t.cpu().numpy()]
        if end_dist is not None:
            end_dist = end_dist[mask_t.cpu().numpy()]
        if haplotypes_t is not None:
            haplotypes_t = haplotypes_t[mask_t]

    # Interaction MAF filter
    if interaction_df is not None:
        genotypes_t, mask_t = filter_maf_interaction(
            genotypes_t, interaction_mask_t=None,
            maf_threshold_interaction=maf_threshold_interaction
        )
        if genotypes_t.shape[0] == 0:
            return None
        mask_np = mask_t.cpu().numpy()
        variant_ids = variant_ids[mask_np]
        start_dist = start_dist[mask_np]
        if end_dist is not None:
            end_dist = end_dist[mask_np]
        if haplotypes_t is not None:
            haplotypes_t = haplotypes_t[mask_t]

    return genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist


def _filter_significant(phenotype_df, phenotype_pos_df, paired_covariate_df,
                        group_s, cis_df, fdr, fdr_col, logger):
    """Subset phenotypes (or groups) to significant associations only."""
    signif_df = cis_df[cis_df[fdr_col] <= fdr].copy()
    if len(signif_df) == 0:
        raise ValueError(f"No significant phenotypes at FDR ≤ {fdr}.")

    cols = [
        'num_var', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df',
        'variant_id', 'start_distance', 'end_distance', 'ma_samples', 'ma_count', 'af',
        'pval_nominal', 'slope', 'slope_se', 'pval_perm', 'pval_beta'
    ]
    if group_s is not None:
        cols += ['group_id', 'group_size']
    signif_df = signif_df[cols]

    if group_s is None:
        ix = phenotype_df.index.intersection(signif_df.index)
        logger.write(f'  * {len(ix)}/{cis_df.shape[0]} significant phenotypes at FDR ≤ {fdr}')
    else:
        ix = group_s[phenotype_df.index].loc[
            group_s[phenotype_df.index].isin(signif_df['group_id'])
        ].index
        logger.write(f'  * {signif_df.shape[0]}/{cis_df.shape[0]} significant groups')
        logger.write(f'    {len(ix)}/{phenotype_df.shape[0]} phenotypes')
        group_s = group_s.loc[ix]

    # Subset inputs
    phenotype_df = phenotype_df.loc[ix]
    phenotype_pos_df = phenotype_pos_df.loc[ix]
    if paired_covariate_df is not None:
        paired_covariate_df = paired_covariate_df.loc[ix]

    return phenotype_df, phenotype_pos_df, paired_covariate_df, group_s, signif_df


def _setup_residualizer_and_interactions(covariates_df, paired_covariate_df,
                                         phenotype_df, interaction_df,
                                         maf_threshold, maf_threshold_interaction,
                                         device, logger):
    """Configure residualizer, dof, and interaction tensor."""
    residualizer = None
    dof = phenotype_df.shape[1] - 2

    if covariates_df is not None:
        logger.write(f'  * {covariates_df.shape[1]} covariates')
        residualizer = Residualizer(_prepare_tensor(covariates_df.values, device=device))
        dof -= covariates_df.shape[1]

    interaction_t = None
    if interaction_df is not None:
        logger.write(f'  * including {interaction_df.shape[1]} interaction term(s)')
        interaction_t = _prepare_tensor(interaction_df.values, device=device)
        dof -= 2 * interaction_df.shape[1]
        if maf_threshold_interaction > 0:
            logger.write(f'    * using {maf_threshold_interaction} MAF threshold')
    elif maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')

    if paired_covariate_df is not None:
        assert covariates_df is not None
        assert paired_covariate_df.index.isin(phenotype_df.index).all(), \
            "Paired covariate phenotypes must be present in phenotype matrix."
        assert paired_covariate_df.columns.equals(phenotype_df.columns), \
            "Paired covariate samples must match phenotype matrix."
        paired_covariate_df = paired_covariate_df.T  # samples × phenotypes
        logger.write(f'  * including phenotype-specific covariate')

    return residualizer, interaction_t, dof, paired_covariate_df


def _setup_mapping_inputs(
        phenotype_df, covariates_df, paired_covariate_df, phenotype_pos_df,
        maf_threshold, interaction_df=None, cis_df=None, fdr=0.05,
        maf_threshold_interaction=0, fdr_col="qval", group_s=None,
        conditional=False, msg=None, logger=None, verbose=True
):
    if logger is None:
        logger = SimpleLogger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.write(f'{msg}')

    signif_df = None
    if conditional:
        assert cis_df is not None, "cis_df must be provided when conditional=True"
        phenotype_df, phenotype_pos_df, paired_covariate_df, group_s, signif_df = \
            _filter_significant(phenotype_df, phenotype_pos_df, paired_covariate_df,
                                group_s, cis_df, fdr, fdr_col, logger)

    assert phenotype_df.index.equals(phenotype_pos_df.index), \
        "phenotype_df and phenotype_pos_df must have matching index"

    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')

    residualizer, interaction_t, dof, paired_covariate_df = \
        _setup_residualizer_and_interactions(
            covariates_df, paired_covariate_df,
            phenotype_df, interaction_df,
            maf_threshold, maf_threshold_interaction,
            device, logger
        )

    return {
        "device": device,
        "logger": logger,
        "residualizer": residualizer,
        "interaction_t": interaction_t,
        "dof": dof,
        "paired_covariate_df": paired_covariate_df,
        "phenotype_df": phenotype_df,
        "phenotype_pos_df": phenotype_pos_df,
        "group_s": group_s,
        "signif_df": signif_df if conditional else None
    }


def _init_result_dict(n, interaction_df, phenotype_pos_df):
    chr_res = OrderedDict()
    chr_res['phenotype_id'] = []
    chr_res['variant_id'] = []
    chr_res['start_distance'] = np.empty(n, dtype=np.int32)
    if 'pos' not in phenotype_pos_df:
        chr_res['end_distance'] = np.empty(n, dtype=np.int32)
    chr_res['af'] = np.empty(n, dtype=np.float32)
    chr_res['ma_samples'] = np.empty(n, dtype=np.int32)
    chr_res['ma_count'] = np.empty(n, dtype=np.int32)

    if interaction_df is None:
        chr_res['pval_nominal'] = np.empty(n, dtype=np.float64)
        chr_res['slope'] = np.empty(n, dtype=np.float32)
        chr_res['slope_se'] = np.empty(n, dtype=np.float32)
    else:
        ni = interaction_df.shape[1]
        chr_res['pval_g'] = np.empty(n, dtype=np.float64)
        chr_res['b_g'] = np.empty(n, dtype=np.float32)
        chr_res['b_g_se'] = np.empty(n, dtype=np.float32)
        chr_res['pval_i'] = np.empty([n, ni], dtype=np.float64)
        chr_res['b_i'] = np.empty([n, ni], dtype=np.float32)
        chr_res['b_i_se'] = np.empty([n, ni], dtype=np.float32)
        chr_res['pval_gi'] = np.empty([n, ni], dtype=np.float64)
        chr_res['b_gi'] = np.empty([n, ni], dtype=np.float32)
        chr_res['b_gi_se'] = np.empty([n, ni], dtype=np.float32)
    return chr_res


def _count_pairs_for_chromosome(igc, chrom, group_s):
    """Count number of phenotype-variant pairs for allocation."""
    n = 0
    if group_s is None:
        for pid in igc.phenotype_pos_df[igc.phenotype_pos_df['chr'] == chrom].index:
            start, end = igc.cis_ranges[pid]
            n += end - start + 1
    else:
        for pid in igc.group_s[igc.phenotype_pos_df['chr'] == chrom].drop_duplicates().index:
            start, end = igc.cis_ranges[pid]
            n += end - start + 1
    return n


def _merge_results(chr_res, chr_block, start, n_i):
    """Insert block of results into chr_res in-place."""
    for k, v in chr_block.items():
        if isinstance(v, list):
            chr_res[k].extend(v)
        else:
            chr_res[k][start:start + n_i] = v


def _prepare_window(row, genotype_ix_t, variant_df, igc, maf_threshold,
                    interaction_df=None, maf_threshold_interaction=0,
                    device=device, logger=logger, warn_monomorphic=True,
                    is_group=False, is_perm=False):
    if not is_group:
        phenotype, genotypes, g_idx, haplotypes, _, phenotype_id = row
        group_id = None
    else:
        phenotypes, genotypes, g_idx, haplotypes, _, phenotype_ids, group_id = row
        phenotype_id = phenotype_ids[0]

    # Prepare tensors
    genotypes_t, haplotypes_t = _prepare_window_tensors(genotypes, haplotypes,
                                                        genotype_ix_t, device)

    # Apply MAF filter
    variant_ids = variant_df.index[g_idx[0]:g_idx[-1] + 1]
    start_dist = variant_df["pos"].values[g_idx] - igc.phenotype_start[phenotype_id]
    end_dist = variant_df["pos"].values[g_idx] - igc.phenotype_end[phenotype_id]
    filt = _apply_maf_filters(genotypes_t, haplotypes_t, variant_ids, start_dist,
                              end_dist, maf_threshold=maf_threshold,
                              interaction_df=interaction_df,
                              maf_threshold_interaction=maf_threshold_interaction)
    if filt is None:
        return None
    genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist = filt

    if is_perm:
        # Filter monomorphic
        genotypes_t, haplotypes_t, g_idx = _filter_monomorphic(
            genotypes_t, haplotypes_t, g_idx, warn=warn_monomorphic, logger=logger
        )
    if genotypes_t.shape[0] == 0:
        label_id = group_id if is_group else phenotype_id
        logger.write(f"WARNING: skipping {label_id} (no valid variants)")
        return None

    new_filt = (genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist)
    if is_group:
        return new_filt, g_idx, phenotypes, phenotype_ids, group_id
    else:
        return new_filter, g_idx, phenotype, phenotype_id


def _write_chromosome_results(
        chr_res, chrom, variant_df, prefix, output_dir,
        interaction_df, logp, paired_covariate_df, mapping_state
):
    """
    Finalizes and writes chromosome-level results to a .parquet file.
    """
    logger = mapping_state["logger"]
    dof = mapping_state["dof"]

    logger.write(f'    Finalizing results for chromosome {chrom}')
    chr_res_df = pd.DataFrame(chr_res)

    # Adjust degrees of freedom if phenotype-specific covariates are used
    if paired_covariate_df is not None:
        idof = dof - chr_res_df['phenotype_id'].isin(paired_covariate_df.index).astype(int).values
    else:
        idof = dof

    # Apply t-to-p conversion
    if interaction_df is None:
        if 'pval_nominal' in chr_res_df.columns:
            mask = chr_res_df['pval_nominal'].notnull()
            chr_res_df.loc[mask, 'pval_nominal'] = get_t_pval(
                chr_res_df.loc[mask, 'pval_nominal'], idof if isinstance(idof, int) else idof[mask], log=logp
            )
    else:
        ni = interaction_df.shape[1]
        if ni == 1:
            for col in ['pval_g', 'pval_i', 'pval_gi']:
                mask = chr_res_df[col].notnull()
                chr_res_df.loc[mask, col] = get_t_pval(
                    chr_res_df.loc[mask, col], idof if isinstance(idof, int) else idof[mask], log=logp
                )
        else:
            # Apply per-interaction column p-value adjustment
            mask = chr_res_df['pval_g'].notnull()
            chr_res_df.loc[mask, 'pval_g'] = get_t_pval(
                chr_res_df.loc[mask, 'pval_g'], idof if isinstance(idof, int) else idof[mask], log=logp
            )
            for i in range(1, ni + 1):
                for suffix in ['pval_i', 'pval_gi']:
                    col = f'{suffix}{i}'
                    if col in chr_res_df.columns:
                        chr_res_df.loc[mask, col] = get_t_pval(
                            chr_res_df.loc[mask, col], idof if isinstance(idof, int) else idof[mask], log=logp
                        )

            # Renaming using interaction names (optional)
            var_dict = {}
            for i, var_name in enumerate(interaction_df.columns, 1):
                for suffix in ['pval_i', 'b_i', 'b_i_se']:
                    var_dict[f"{suffix}{i}"] = f"{suffix[:-1]}_{var_name}"
                for suffix in ['pval_gi', 'b_gi', 'b_gi_se']:
                    var_dict[f"{suffix}{i}"] = f"{suffix[:-2]}_g-{var_name}"
            chr_res_df.rename(columns=var_dict, inplace=True)

    # Write to disk
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{prefix}.cis_qtl_pairs.{chrom}.parquet")
    try:
        chr_res_df.to_parquet(outfile, index=False)
        logger.write(f'    * wrote {outfile}')
    except Exception as e:
        logger.write(f'    ! Failed to write {outfile}: {e}')
        raise


def _summarize_top_associations(
        best_assoc, interaction_df, logp, run_eigenmt, write_top,
        output_dir, prefix, mapping_state, group_s=None
):
    """
    Combine top associations across chromosomes and write/return summary.
    """
    logger = mapping_state["logger"]
    dof = mapping_state["dof"]
    paired_covs_df = mapping_state["paired_covariate_df"]

    logger.write("  * Summarizing top associations")

    # Combine top hits
    top_df = pd.concat(best_assoc, axis=1, sort=False).T
    top_df.index = top_df['phenotype_id']
    top_df = top_df.infer_objects()

    ni = interaction_df.shape[1]

    # Convert t-stats to p-values
    m = top_df['pval_g'].notnull()
    m_idx = top_df[m].index

    if paired_covs_df is not None:
        idof = dof - top_df.index.isin(paired_covs_df.index).astype(int)
        idof = idof[m_idx]
    else:
        idof = dof

    # Apply p-value transformation
    top_df.loc[m_idx, 'pval_g'] = get_t_pval(top_df.loc[m_idx, 'pval_g'], idof, log=logp)

    if ni == 1:
        top_df.loc[m_idx, 'pval_i'] = get_t_pval(top_df.loc[m_idx, 'pval_i'], idof, log=logp)
        top_df.loc[m_idx, 'pval_gi'] = get_t_pval(top_df.loc[m_idx, 'pval_gi'], idof, log=logp)
    else:
        for i in range(1, ni + 1):
            top_df.loc[m_idx, f'pval_i{i}'] = get_t_pval(top_df.loc[m_idx, f'pval_i{i}'], idof, log=logp)
            top_df.loc[m_idx, f'pval_gi{i}'] = get_t_pval(top_df.loc[m_idx, f'pval_gi{i}'], idof, log=logp)

        # Rename interaction columns using column names from interaction_df
        var_dict = {}
        for i, v in enumerate(interaction_df.columns, 1):
            for col in ['pval_i', 'b_i', 'b_i_se']:
                var_dict[f'{col}{i}'] = f'{col[:-1]}_{v}'
            for col in ['pval_gi', 'b_gi', 'b_gi_se']:
                var_dict[f'{col}{i}'] = f'{col[:-2]}_g-{v}'
        top_df.rename(columns=var_dict, inplace=True)

    # Apply eigenMT if enabled
    if run_eigenmt and ni == 1:
        if group_s is None:
            top_df['pval_emt'] = np.minimum(top_df['tests_emt'] * top_df['pval_gi'], 1.0)
        else:
            top_df['pval_emt'] = np.minimum(
                top_df['num_phenotypes'] * top_df['tests_emt'] * top_df['pval_gi'], 1.0
            )
        top_df['pval_adj_bh'] = eigenmt.padjust_bh(top_df['pval_emt'])

    # Output or return
    outfile = os.path.join(output_dir, f'{prefix}.cis_qtl_top_assoc.txt.gz')
    if write_top:
        top_df.to_csv(outfile, sep='\t', float_format='%.6g')
        logger.write(f'  * wrote top associations: {outfile}')
    else:
        return top_df
