import torch
import numpy as np
import pandas as pd

from core import (
    Residualizer,
    prepare_cis_output,
    calculate_cis_permutations,
    calculate_beta_approx_pval
)
from utils import _prepare_tensor, _prepare_window

def _forward_pass(
        phenotype_t, genotypes_t, haplotypes_t, permutation_ix_t,
        covariates, dof, g_idx, variant_df, phenotype_id, igc, nperm,
        signif_threshold, logp, random_tiebreak, device
):
    """Run one forward pass step: add new variant, rerun permutations, check significance."""
    residualizer = Residualizer(
        _prepare_tensor(covariates, device=device)
    )

    res = calculate_cis_permutations(
        genotypes_t, phenotype_t, permutation_ix_t,
        residualizer=residualizer, haplotypes_t=haplotypes_t,
        random_tiebreak=random_tiebreak
    )
    r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
    x = calculate_beta_approx_pval(r2_perm, r_nominal * r_nominal, dof)

    if x[0] <= signif_threshold:
        var_ix = g_idx[var_ix]
        variant_id = variant_df.index[var_ix]
        start_distance = variant_df["pos"].values[var_ix] - igc.phenotype_start[phenotype_id]
        end_distance = variant_df["pos"].values[var_ix] - igc.phenotype_end[phenotype_id]
        res_s = prepare_cis_output(
            r_nominal, r2_perm, std_ratio, g,
            genotypes_t.shape[0], dof, variant_id,
            start_distance, end_distance, phenotype_id,
            nperm=nperm, logp=logp
        )
        res_s[["pval_beta", "beta_shape1", "beta_shape2", "true_df", "pval_true_df"]] = x
        return res_s
    return None


def _backward_pass(
        forward_df, dosage_df, genotypes_t, haplotypes_t, permutation_ix_t,
        covariates_df, g_idx, variant_df, igc, nperm, signif_threshold, logp,
        random_tiebreak, device, phenotypes=None, phenotype_ids=None, group_id=None
):
    """
    Backward pass: test each variant by leaving it out.
    Works for single phenotypes or grouped phenotypes.
    """
    back_df, variant_set = [], set()

    for k, variant_id in enumerate(forward_df["variant_id"], 1):
        # Build covariates with all but current variant
        if covariates_df is not None:
            covariates = np.hstack([
                covariates_df.values,
                dosage_df[np.setdiff1d(forward_df["variant_id"], variant_id)].values,
            ])
        else:
            covariates = dosage_df[np.setdiff1d(forward_df["variant_id"], variant_id)].values
        dof = genotypes_t.shape[1] - 2 - covariates.shape[1]  # n_samples - 2 - covariates

        # Single phenotype
        if phenotypes is None:
            phenotype_id = forward_df.index[0] if "phenotype_id" in forward_df else None
            phenotype_t = _prepare_tensor(forward_df.loc[phenotype_id], device=device)
            res_s = _forward_pass(
                phenotype_t, genotypes_t, haplotypes_t, permutation_ix_t,
                covariates, dof, g_idx, variant_df, phenotype_id,
                igc, nperm, logp, random_tiebreak, device, signif_threshold
            )
            if res_s is not None and res_s["variant_id"] not in variant_set:
                res_s["rank"] = k
                back_df.append(res_s)
                variant_set.add(res_s["variant_id"])

        # Grouped phenotype
        else:
            buf = []
            for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
                phenotype_t = _prepare_tensor(phenotype, device=device)
                res_s = _forward_pass(
                    phenotype_t, genotypes_t, haplotypes_t, permutation_ix_t,
                    covariates, dof, g_idx, variant_df, phenotype_id,
                    igc, nperm, logp, random_tiebreak, device, signif_threshold
                )
                if res_s is not None:
                    buf.append(res_s)
            if len(buf) > 0:
                res_s = _process_group_permutations(
                    buf, variant_df,
                    igc.phenotype_start[phenotype_ids[0]],
                    igc.phenotype_end[phenotype_ids[0]],
                    dof, group_id, nperm=nperm, logp=logp
                )
                if res_s["pval_beta"] <= signif_threshold and variant_id not in variant_set:
                    res_s["rank"] = k
                    back_df.append(res_s)
                    variant_set.add(variant_id)

    return pd.concat(back_df, axis=1, sort=False).T if back_df else None


def _process_independent_window(
        row, genotype_ix_t, variant_df, genotype_df, ix_dict, igc,
        mapping_state, permutation_ix_t, covariates_df, nperm,
        signif_threshold, maf_threshold, missing, random_tiebreak, logp):
    """
    Forward-backward independent cis-QTL mapping for a single phenotype window.
    Returns a DataFrame of independent variants or None.
    """
    device, logger = mapping_state["device"], mapping_state["logger"]
    filt, g_idx, phenotype, phenotype_id = _prepare_window(
        row, genotype_ix_t, variant_df, igc, maf_threshold,
        device=device, logger=logger, is_group=False, is_perm=False
    )

    if filt is None:
        return None
    genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist = filt

    # Phenotype tensor
    phenotype_t = _prepare_tensor(phenotype, device=device)

    # Forward pass
    forward_df = [mapping_state["signif_df"].loc[phenotype_id]]  # seed with top variant
    covariates = covariates_df.values.copy() if covariates_df is not None else np.empty((phenotype_t.shape[0], 0))
    dosage_dict = {}

    while True:
        variant_id = forward_df[-1]["variant_id"]
        ig = genotype_df.values[ix_dict[variant_id], :].copy()
        m = ig == missing
        ig[m] = ig[~m].mean()
        dosage_dict[variant_id] = ig
        covariates = np.hstack([covariates, ig.reshape(-1, 1)]).astype(np.float32)
        dof = phenotype_t.shape[0] - 2 - covariates.shape[1]

        res_s = _forward_pass(
            phenotype_t, genotypes_t, haplotypes_t, permutation_ix_t,
            covariates, dof, g_idx, variant_df, phenotype_id,
            igc, nperm, signif_threshold, logp, random_tiebreak, device
        )
        if res_s is not None:
            forward_df.append(res_s)
        else:
            break

    forward_df = pd.concat(forward_df, axis=1, sort=False).T
    dosage_df = pd.DataFrame(dosage_dict)

    # Backward pass
    if forward_df.shape[0] > 1:
        back_df = _backward_pass(
            forward_df, dosage_df, phenotype_t, genotypes_t,
            haplotypes_t, permutation_ix_t, covariates_df,
            g_idx, variant_df, phenotype_id, igc, nperm,
            signif_threshold, logp, random_tiebreak, device
        )
        return back_df if back_df is not None else None
    else:
        forward_df["rank"] = 1
        return forward_df


def _process_group_independent_window(
        row, genotype_ix_t, variant_df, genotype_df, ix_dict, igc, mapping_state,
        permutation_ix_t, covariates_df, nperm, signif_threshold,
        maf_threshold, missing, random_tiebreak, logp):
    """
    Forward-backward independent cis-QTL mapping for grouped phenotypes.
    Returns a DataFrame of independent variants or None.
    """
    device, logger = mapping_state["device"], mapping_state["logger"]

    filt, g_idx, phenotypes, phenotype_ids, group_id = _prepare_window(
        row, genotype_ix_t, variant_df, igc, maf_threshold,
        device=device, logger=logger, is_group=True, is_perm=False
    )

    if filt is None:
        return None
    genotypes_t, haplotypes_t, variant_ids, start_dist, end_dist = filt

    # Forward pass
    forward_df = [mapping_state["signif_df"][mapping_state["signif_df"]["group_id"] == group_id].iloc[0]]
    covariates = covariates_df.values.copy() if covariates_df is not None else np.empty((phenotypes.shape[1], 0))
    dosage_dict = {}

    while True:
        variant_id = forward_df[-1]["variant_id"]
        ig = genotype_df.values[ix_dict[variant_id], :].copy()
        m = ig == missing
        ig[m] = ig[~m].mean()
        dosage_dict[variant_id] = ig
        covariates = np.hstack([covariates, ig.reshape(-1, 1)]).astype(np.float32)
        dof = phenotypes.shape[1] - 2 - covariates.shape[1]

        # run across all phenotypes in group
        buf = []
        for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
            phenotype_t = _prepare_tensor(phenotype, device=device)
            res_s = _forward_pass(
                phenotype_t, genotypes_t, haplotypes_t, permutation_ix_t,
                covariates, dof, g_idx, variant_df, phenotype_id,
                igc, nperm, signif_threshold, logp, random_tiebreak, device
            )
            if res_s is not None:
                buf.append(res_s)

        if len(buf) > 0:
            res_s = _process_group_permutations(
                buf, variant_df,
                igc.phenotype_start[phenotype_ids[0]],
                igc.phenotype_end[phenotype_ids[0]],
                dof, group_id,
                nperm=nperm, logp=logp
            )
            if res_s["pval_beta"] <= signif_threshold:
                forward_df.append(res_s)
            else:
                break
        else:
            break

    forward_df = pd.concat(forward_df, axis=1, sort=False).T
    dosage_df = pd.DataFrame(dosage_dict)

    # Backward pass
    if forward_df.shape[0] > 1:
        back_df = _backward_pass(
            forward_df, dosage_df, genotypes_t, haplotypes_t, permutation_ix_t,
            covariates_df, g_idx, variant_df, igc, nperm, signif_threshold,
            logp, random_tiebreak, device, phenotypes=phenotypes,
            phenotype_ids=phenotype_ids, group_id=group_id
        )
        return back_df if back_df is not None else None
    else:
        forward_df["rank"] = 1
        return forward_df
