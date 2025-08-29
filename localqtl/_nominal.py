import time
import torch
import numpy as np
import pandas as pd

import eigenmt
from core import (
    Residualizer,
    calculate_cis_nominal,
    calculate_interaction_nominal
)

from utils import (
    _merge_results,
    _prepare_tensor,
    _init_result_dict,
    _apply_maf_filters,
    _prepare_window_tensors,
    _count_pairs_for_chromosome
)

def _run_association(genotypes_t, phenotype_t, haplotypes_t,
                     residualizer, interaction_df, interaction_t,
                     variant_ids, device):
    """Run cis or interaction association depending on interaction_df."""
    if interaction_df is None:
        res = calculate_cis_nominal(
            genotypes_t, phenotype_t, residualizer=residualizer,
            haplotypes_t=haplotypes_t
        )
        return [x.cpu().numpy() for x in res], None
    else:
        res = calculate_interaction_nominal(
            genotypes_t, phenotype_t.unsqueeze(0), interaction_t,
            residualizer=residualizer,
            haplotypes_t=haplotypes_t,
            variant_ids=variant_ids,
            return_sparse=False
        )
        return [x.cpu().numpy() for x in res], interaction_df.shape[1]


def _map_chromosome(chrom, igc, variant_df, phenotype_pos_df, mapping_state,
                    group_s, genotype_df, sample_ids, maf_threshold, interaction_df,
                    maf_threshold_interaction, logp, run_eigenmt, verbose,
                    start_time, covariates_df):
    """
    Map cis-QTLs for a single chromosome.

    Returns:
        - chr_res: dict of association results
        - top_hits: list of top association Series (if interaction_df provided)
    """
    device = mapping_state['device']
    residualizer = mapping_state['residualizer']
    interaction_t = mapping_state['interaction_t']
    paired_covs = mapping_state['paired_covariate_df']
    logger = mapping_state['logger']

    logger.write(f'    Mapping chromosome {chrom}')

    # Preallocate results
    n = _count_pairs_for_chromosome(igc, chrom, group_s)
    chr_res = _init_result_dict(n, interaction_df, phenotype_pos_df)
    best_assoc = []

    start = 0
    genotype_ix = np.array([genotype_df.columns.get_loc(s) for s in sample_ids])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # Iterate windows
    for row in igc.generate_data(chrom=chrom, verbose=verbose):
        process_fnc = _process_grouped_phenotype_window if group_s is not None else _process_phenotype_window
        results = process_fnc(
            row, genotype_ix_t, variant_df, phenotype_pos_df,
            covariates_df, residualizer, paired_covs, interaction_t,
            maf_threshold, interaction_df, maf_threshold_interaction,
            run_eigenmt, device
        )
        if results is None:
            continue

        n_i, chr_block, top_hit = results
        _merge_results(chr_res, chr_block, start, n_i)

        if top_hit is not None:
            best_assoc.append(top_hit)
        start += n_i

    logger.write(f'    time elapsed: {(time.time()-start_time)/60:.2f} min')

    # Clip any unused preallocated array space
    for k in chr_res:
        if not isinstance(chr_res[k], list):
            chr_res[k] = chr_res[k][:start]

    return chr_res, best_assoc


def _process_phenotype_window(
        row, genotype_ix_t, variant_df, phenotype_pos_df,
        covariates_df, residualizer, paired_covs_df, interaction_t,
        maf_threshold, interaction_df, maf_threshold_interaction,
        run_eigenmt, device):
    """
    Process one cis-window for a phenotype (or group of phenotypes).

    Returns:
        - n: number of variants analyzed
        - result_dict: dictionary of results (same structure as chr_res block)
        - top_hit: Series with top association (or None)
    """
    phenotypes, genotypes, g_idx, haplotypes, _, phenotype_ids = row

    variant_ids = variant_df.index[g_idx[0]:g_idx[-1] + 1]
    start_dist = variant_df['pos'].values[g_idx[0]:g_idx[-1] + 1] - \
                 igc.phenotype_start[phenotype_id]
    end_dist = None
    if 'pos' not in phenotype_pos_df:
        end_dist = variant_df['pos'].values[g_idx[0]:g_idx[-1] + 1] - \
                   igc.phenotype_end[phenotype_id]

    genotypes_t, haplotypes_t = _prepare_window_tensors(genotypes, haplotypes,
                                                        genotype_ix_t, device)
    filt = _apply_maf_filters(genotypes_t, haplotypes_t, variant_ids, start_dist,
                              end_dist, maf_threshold, interaction_df,
                              maf_threshold_interaction)
    if filt is None:
        return None
    genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist = filt

    # Residualizer (with optional phenotype-specific covariate)
    if paired_covs_df is not None and phenotype_id in paired_covs_df.index:
        pcov_t = _prepare_tensor(np.c_[covariates_df,
                                       paired_covs_df[phenotype_id]],
                                 device=device)
        iresidualizer = Residualizer(pcov_t)
    else:
        iresidualizer = residualizer

    # Run association model
    phenotype_t = _prepare_tensor(phenotype, device=device)
    results, ni = _run_association(genotypes_t, phenotype_t, haplotypes_t,
                                   iresidualizer, interaction_df, interaction_t,
                                   variant_ids, device)
    if interaction_df is None:
        tstat, slope, slope_se, af, ma_samples, ma_count = results
        result = dict(
            phenotype_id=[phenotype_id] * len(variant_ids),
            variant_id=variant_ids,
            start_distance=start_dist,
            af=af, ma_samples=ma_samples, ma_count=ma_count,
            pval_nominal=tstat, slope=slope, slope_se=slope_se
        )
        if end_dist is not None:
            result['end_distance'] = end_dist
        return len(variant_ids), result, None
    else:
        tstat, b, b_se, af, ma_samples, ma_count = results
        ni = interaction_df.shape[1]
        result = dict(
            phenotype_id=[phenotype_id] * len(variant_ids),
            variant_id=variant_ids,
            start_distance=start_dist,
            af=af, ma_samples=ma_samples, ma_count=ma_count,
            pval_g=tstat[:, 0], b_g=b[:, 0], b_g_se=b_se[:, 0],
            pval_i=tstat[:, 1:1 + ni], b_i=b[:, 1:1 + ni], b_i_se=b_se[:, 1:1 + ni],
            pval_gi=tstat[:, 1 + ni:], b_gi=b[:, 1 + ni:], b_gi_se=b_se[:, 1 + ni:]
        )
        if end_dist is not None:
            result['end_distance'] = end_dist

        # Top association
        ix = np.nanargmax(np.abs(tstat[:, 1 + ni:]).max(1))
        top = dict(
            phenotype_id=phenotype_id,
            variant_id=variant_ids[ix],
            start_distance=start_dist[ix],
            af=af[ix], ma_samples=ma_samples[ix], ma_count=ma_count[ix],
        )
        if end_dist is not None:
            top['end_distance'] = end_dist[ix]
        for i in range(tstat.shape[1]):
            top[f'stat_{i}'] = tstat[ix, i]
            top[f'beta_{i}'] = b[ix, i]
            top[f'se_{i}'] = b_se[ix, i]
        if run_eigenmt:
            top['tests_emt'] = eigenmt.compute_tests(genotypes_t)

        return len(variant_ids), result, pd.Series(top)


def _process_grouped_phenotype_window(
        row, genotype_ix_t, variant_df, phenotype_pos_df,
        covariates_df, residualizer, paired_covs_df, interaction_t,
        maf_threshold, interaction_df, maf_threshold_interaction,
        run_eigenmt, device
):
    """
    Process one cis-window for a group of phenotypes (group_s is not None).

    Returns:
        - n: number of variants analyzed
        - result_dict: dictionary of results (same structure as chr_res block)
        - top_hit: Series with top association (or None)
    """
    phenotypes, genotypes, g_idx, haplotypes, _, phenotype_ids, group_id = row
    variant_ids = variant_df.index[g_idx[0]:g_idx[-1] + 1]
    start_dist = variant_df['pos'].values[g_idx[0]:g_idx[-1] + 1] - \
                 igc.phenotype_start[phenotype_ids[0]]
    end_dist = None
    if 'pos' not in phenotype_pos_df:
        end_dist = variant_df['pos'].values[g_idx[0]:g_idx[-1] + 1] - \
                   igc.phenotype_end[phenotype_ids[0]]

    genotypes_t, haplotypes_t = _prepare_window_tensors(genotypes, haplotypes,
                                                        genotype_ix_t, device)
    filt = _apply_maf_filters(genotypes_t, haplotypes_t, variant_ids, start_dist,
                              end_dist, maf_threshold, interaction_df,
                              maf_threshold_interaction)
    if filt is None:
        return None
    genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist = filt

    # Run for first phenotype
    phenotype_t = _prepare_tensor(phenotypes[0], device=device)
    results, _ = _run_association(genotypes_t, phenotype_t, haplotypes_t,
                                  residualizer, interaction_df, interaction_t,
                                  variant_ids, device)

    if interaction_df is None:
        tstat, slope, slope_se, af, ma_samples, ma_count = results
    else:
        tstat, b, b_se, af, ma_samples, ma_count = results

    px = [phenotype_ids[0]] * len(variant_ids)

    # Iterate over remaining phenotypes and update stronger associations
    for phenotype, pid in zip(phenotypes[1:], phenotype_ids[1:]):
        phenotype_t = _prepare_tensor(phenotype, device=device)
        results, _ = _run_association(genotypes_t, phenotype_t, haplotypes_t,
                                      residualizer, interaction_df, interaction_t,
                                      variant_ids, device)
        if interaction_df is None:
            tstat0, slope0, slope_se0, _, _, _ = results
            ix = np.where(np.abs(tstat0) > np.abs(tstat))[0]
            tstat[ix] = tstat0[ix]; slope[ix] = slope0[ix]; slope_se[ix] = slope_se0[ix]
        else:
            tstat0, b0, b_se0, _, _, _ = results
            ix = np.where(np.abs(tstat0[:, 2]) > np.abs(tstat[:, 2]))[0]
            tstat[ix] = tstat0[ix]; b[ix] = b0[ix]; b_se[ix] = b_se0[ix]
        for j in ix: px[j] = pid

    # Build results
    result = dict(
        phenotype_id=px,
        variant_id=variant_ids,
        start_distance=start_dist,
        af=af, ma_samples=ma_samples, ma_count=ma_count,
    )
    if end_dist is not None:
        result['end_distance'] = end_dist

    if interaction_df is None:
        result.update(dict(pval_nominal=tstat, slope=slope, slope_se=slope_se))
        return len(variant_ids), result, None
    else:
        ni = interaction_df.shape[1]
        result.update(dict(
            pval_g=tstat[:, 0], b_g=b[:, 0], b_g_se=b_se[:, 0],
            pval_i=tstat[:, 1:1 + ni], b_i=b[:, 1:1 + ni], b_i_se=b_se[:, 1:1 + ni],
            pval_gi=tstat[:, 1 + ni:], b_gi=b[:, 1 + ni:], b_gi_se=b_se[:, 1 + ni:]
        ))
        # Top association within group
        ix = np.nanargmax(np.abs(tstat[:, 1 + ni:]).max(1))
        top = dict(
            phenotype_id=result['phenotype_id'][ix],
            variant_id=variant_ids[ix],
            start_distance=start_dist[ix],
            af=af[ix], ma_samples=ma_samples[ix], ma_count=ma_count[ix],
            num_phenotypes=len(phenotype_ids)
        )
        if end_dist is not None:
            top['end_distance'] = end_dist[ix]
        for i in range(tstat.shape[1]):
            top[f'stat_{i}'] = tstat[ix, i]
            top[f'beta_{i}'] = b[ix, i]
            top[f'se_{i}'] = b_se[ix, i]
        if run_eigenmt:
            top['tests_emt'] = eigenmt.compute_tests(genotypes_t)

    return len(variant_ids), result, pd.Series(top)
