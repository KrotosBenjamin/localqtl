import time
import torch
import numpy as np
import pandas as pd

import eigenmt
from core import (
    Residualizer,
    calculate_cis_nominal,
    calculate_corr_paired,
    calculate_interaction_nominal
)

from utils import (
    _merge_results,
    _prepare_tensor,
    _batch_generator,
    _init_result_dict,
    _apply_maf_filters,
    _unpack_hap_effects,
    _prepare_window_tensors,
    _count_pairs_for_chromosome
)

def _run_association(genotypes_t, phenotype_t, haplotypes_t,
                     residualizer, interaction_df, interaction_t,
                     variant_ids, device, dof_vector):
    """Run cis or interaction association depending on interaction_df."""
    if interaction_df is None:
        if phenotype_t.shape[0] == 1:
            res = calculate_cis_nominal(
                genotypes_t, phenotype_t,
                residualizer=residualizer,
                haplotypes_t=haplotypes_t,
            )
        else:
            res = calculate_corr_paired(
                genotypes_t, haplotypes_t, phenotype_t,
                residualizer=None, use_pinv=False,
                return_se_h=True, dof_vector=dof_vector,
            )
        return [x.cpu().numpy() for x in res], None
    else:
        # TODO: batch model
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
                    start_time, covariates_df, start=0):
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

    genotype_ix = np.array([genotype_df.columns.get_loc(s) for s in sample_ids])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # Iterate windows
    for batch_idx, batch_rows in enumerate(_batch_generator(
            igc.generate_data(chrom=chrom, verbose=verbose),
            device=mapping_state["device"])):
        process_fnc = _process_grouped_phenotype_window if group_s is not None else _process_phenotype_window
        results = process_fnc(
            batch_rows, igc, genotype_ix_t, variant_df, phenotype_pos_df,
            covariates_df, residualizer, paired_covs, interaction_t,
            maf_threshold, interaction_df, maf_threshold_interaction,
            run_eigenmt, mapping_state
        )
        if results is None:
            continue

        n_i, chr_block, top_hit = results
        _merge_results(chr_res, chr_block, start, n_i)

        if top_hit is not None:
            best_assoc.append(top_hit)
        start += n_i

        del batch_rows, results, chr_block
        if torch.cuda.is_available() and batch_idx % 250 == 0:
            torch.cuda.empty_cache()

    logger.write(f'    time elapsed: {(time.time()-start_time)/60:.2f} min')

    # Clip any unused preallocated array space
    for k in chr_res:
        if not isinstance(chr_res[k], list):
            chr_res[k] = chr_res[k][:start]

    return chr_res, best_assoc, start


def _process_phenotype_window(
        rows, igc, genotype_ix_t, variant_df, phenotype_pos_df,
        covariates_df, residualizer, paired_covs_df, interaction_t,
        maf_threshold, interaction_df, maf_threshold_interaction,
        run_eigenmt, mapping_state):
    """
    Process one cis-window *batch*:
      - stacks all phenotype windows together (z-stack),
      - runs calculate_corr_paired once,
      - returns results ready for merge.

    Returns
    -------
    n_total : int
        Total number of variant–phenotype pairs
    merged : dict
        Block of results (same keys as chr_res)
    top_hit : None for now (interaction not supported yet)
    """
    device = mapping_state["device"]

    geno_list, hap_list, pheno_list = [], [], []
    varid_list, vardist_list, vardist_end_list, varidx_list = [], [], [], []
    dof_list = []
    phenotype_ids = []

    for phenotype, genotypes, g_idx, haplotypes, phenotype_id in rows:
        variant_idx = g_idx
        variant_ids = variant_df.index[g_idx]
        variant_pos = variant_df['pos'].to_numpy(copy=False)

        start_dist = variant_pos[g_idx] - igc.phenotype_start[phenotype_id]
        end_dist = None
        if 'pos' not in phenotype_pos_df:
            end_dist = variant_pos[g_idx] - igc.phenotype_end[phenotype_id]

        # Slice tensors
        G_t, H_t = _prepare_window_tensors(genotypes, haplotypes, genotype_ix_t, device)
        phenotype_t = _prepare_tensor(phenotype, device=device)

        # MAF filters
        filt = _apply_maf_filters(
            G_t, H_t, variant_ids, start_dist, end_dist,
            maf_threshold, interaction_df,
            maf_threshold_interaction, variant_idx, mapping_state
        )
        if filt is None:
            continue
        G_t, H_t, variant_ids, start_dist, end_dist, variant_idx = filt

        # Residualizer (with optional phenotype-specific covariates)
        if paired_covs_df is not None and phenotype_id in paired_covs_df.index:
            pcov_t = _prepare_tensor(
                np.c_[covariates_df, paired_covs_df.loc[[phenotype_id]].values.T],
                device=device
            )
            iresid = Residualizer(pcov_t)
        else:
            iresid = residualizer

        # Residualize
        if iresid is not None:
            G_t = iresid.transform(G_t)
            phenotype_t = iresid.transform(phenotype_t.unsqueeze(0)).squeeze(0)
            if H_t is not None:
                if H_t.ndim == 2:
                    H_t = iresid.transform(H_t).unsqueeze(-1)
                elif H_t.ndim == 3:
                    n_var, n_samp, k = H_t.shape
                    H_flat = H_t.reshape(n_var, n_samp * k)
                    H_t = iresid.transform(H_flat).reshape(n_var, n_samp, k)

        if H_t is not None and H_t.ndim == 2:
            H_t = H_t.unsqueeze(-1)
        # Collect
        geno_list.append(G_t)
        hap_list.append(H_t)
        pheno_list.append(phenotype_t.repeat(G_t.shape[0], 1))
        varid_list.append(variant_ids)
        vardist_list.append(start_dist)
        vardist_end_list.append(end_dist)
        varidx_list.append(variant_idx)
        phenotype_ids.append(phenotype_id)

        # DOF per variant in this window
        n_samples = G_t.shape[1]
        n_covs = covariates_df.shape[1] if covariates_df is not None else 0
        k = H_t.shape[2] if H_t is not None else 0
        extra_covs = 1 if (paired_covs_df is not None and phenotype_id in paired_covs_df.index) else 0
        dof_val = n_samples - (n_covs + extra_covs + 1 + k)
        dof_list.append(np.full(G_t.shape[0], dof_val, dtype=np.float32))

    if not geno_list:
        return None

    # --- Z-stack all windows ---
    G_all = torch.cat(geno_list, dim=0)             # (Σv, s)
    H_all = torch.cat([h for h in hap_list if h is not None], dim=0) if any(h is not None for h in hap_list) else None
    Y_all = torch.cat(pheno_list, dim=0)            # (Σv, s)
    dof_vector = torch.from_numpy(np.concatenate(dof_list)).to(device)

    # --- Run regression once ---
    beta_g, beta_h, tstat_g, se_g, se_h = calculate_corr_paired(
        G_all, H_all, Y_all, residualizer=None, dof_vector=dof_vector
    )

    # --- Build results back into flat dict ---
    results = []
    offset = 0
    for pid, variant_ids, start_dist, end_dist, variant_idx, n_vars in zip(
            phenotype_ids, varid_list, vardist_list, vardist_end_list, varidx_list,
            [g.shape[0] for g in geno_list]):
        sl = slice(offset, offset + n_vars)
        result = dict(
            phenotype_id=[pid] * n_vars,
            variant_id=variant_ids,
            start_distance=start_dist,
            af=mapping_state["af_all"][variant_idx],
            ma_samples=mapping_state["ma_samples_all"][variant_idx],
            ma_count=mapping_state["ma_count_all"][variant_idx],
            pval_nominal=tstat_g[sl].cpu().numpy(),
            beta_g=beta_g[sl].cpu().numpy(),
            se_g=se_g[sl].cpu().numpy(),
            **_unpack_hap_effects(beta_h[sl].cpu().numpy(), se_h[sl].cpu().numpy())
        )
        if end_dist is not None:
            result["end_distance"] = end_dist
        results.append(result)
        offset += n_vars

    merged = {k: np.concatenate([r[k] for r in results], axis=0)
              for k in results[0]}
    n_total = offset
    ## TODO: interaction model
    return n_total, merged, None


def _process_grouped_phenotype_window(
        row, igc, genotype_ix_t, variant_df, phenotype_pos_df,
        covariates_df, residualizer, paired_covs_df, interaction_t,
        maf_threshold, interaction_df, maf_threshold_interaction,
        run_eigenmt, mapping_state
):
    """
    Process one cis-window for a group of phenotypes (group_s is not None).

    Returns:
        - n: number of variants analyzed
        - result_dict: dictionary of results (same structure as chr_res block)
        - top_hit: Series with top association (or None)
    """
    device = mapping_state["device"]
    phenotypes, genotypes, g_idx, haplotypes, phenotype_ids, group_id = row

    variant_idx = g_idx
    variant_ids = variant_df.index[g_idx[0]:g_idx[-1] + 1]
    variant_pos = variant_df['pos'].to_numpy(copy=False)
    start_dist = variant_pos[g_idx[0]:g_idx[-1] + 1] - igc.phenotype_start[phenotype_ids[0]]
    end_dist = None
    if 'pos' not in phenotype_pos_df:
        end_dist = variant_pos[g_idx[0]:g_idx[-1] + 1] - igc.phenotype_end[phenotype_ids[0]]

    G_t, H_t = _prepare_window_tensors(genotypes, haplotypes,
                                                        genotype_ix_t, device)
    filt = _apply_maf_filters(G_t, H_t, variant_ids, start_dist,
                              end_dist, maf_threshold, interaction_df,
                              maf_threshold_interaction, variant_idx, mapping_state)
    if filt is None:
        return None
    G_t, H_t, variant_ids, start_dist, end_dist, variant_idx = filt

    # Run for first phenotype
    phenotype_t = _prepare_tensor(phenotypes[0], device=device)
    results, _ = _run_association(G_t, phenotype_t, H_t,
                                  residualizer, interaction_df, interaction_t,
                                  variant_ids, device)

    if interaction_df is None:
        tstat, beta_g, se_g, beta_h, se_h = results
        af = mapping_state["af_all"][variant_idx]
        ma_samples = mapping_state["ma_samples_all"][variant_idx]
        ma_count = mapping_state["ma_count_all"][variant_idx]
    else:
        tstat, b, b_se, af, ma_samples, ma_count = results

    px = [phenotype_ids[0]] * len(variant_ids)

    # Iterate over remaining phenotypes and update stronger associations
    for phenotype, pid in zip(phenotypes[1:], phenotype_ids[1:]):
        phenotype_t = _prepare_tensor(phenotype, device=device)
        results, _ = _run_association(G_t, phenotype_t, H_t,
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
            top['tests_emt'] = eigenmt.compute_tests(G_t)

    return len(variant_ids), result, pd.Series(top)
