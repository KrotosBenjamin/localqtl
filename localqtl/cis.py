import torch
import numpy as np
import pandas as pd
import sys, os, time

sys.path.insert(1, os.path.dirname(__file__))
import local
from utils import (
    _init_result_dict,
    _log_mapping_context,
    _setup_mapping_inputs,
    _write_chromosome_results,
    _summarize_top_associations
)
from _nominal import (
    _map_chromosome,
    _process_phenotype_window,
    _process_grouped_phenotype_window
)
from _permutation import (
    _make_permutation_index,
    _process_permutation_window,
    _process_group_permutation_window
)
from _independent import (
    _process_independent_window,
    _process_group_independent_window
)

def map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, prefix,
                haplotype_df=None, loci_df=None, covariates_df=None,
                paired_covariate_df=None, maf_threshold=0, interaction_df=None,
                maf_threshold_interaction=0.05, group_s=None, window=1_000_000,
                run_eigenmt=False, logp=False, output_dir='.', write_top=True,
                write_stats=True, logger=None, verbose=True):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet

    If interaction_df is provided, the top association per phenotype is
    written to <output_dir>/<prefix>.cis_qtl_top_assoc.txt.gz unless
    write_top is set to False, in which case it is returned as a DataFrame
    """
    msg='cis-QTL mapping: nominal associations for all variant-phenotype pairs'
    mapping_state = _setup_mapping_inputs(
        phenotype_df, covariates_df, paired_covariate_df, phenotype_pos_df,
        maf_threshold, interaction_df=interaction_df,
        maf_threshold_interaction=maf_threshold_interaction, logger=logger,
        group_s=group_s, msg=msg, verbose=verbose
    )
    phenotype_df = mapping_state["phenotype_df"]
    phenotype_pos_df = mapping_state["phenotype_pos_df"]
    group_s = mapping_state["group_s"]
    sample_ids = phenotype_df.columns.tolist()

    _log_mapping_context(
        genotype_df, haplotype_df, group_s, window,
        None, mapping_state["logger"]
    )

    igc = local.InputGeneratorCis(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        loci_df if loci_df is not None else pd.DataFrame(index=[]),
        haplotype_df if haplotype_df is not None else pd.DataFrame(index=[]),
        group_s=group_s, window=window
    )

    best_assoc = []
    start_time = time.time()
    for chrom in igc.chrs:
        chr_res, top_hits = _map_chromosome(
            chrom, igc, variant_df, phenotype_pos_df, mapping_state,
            group_s, genotype_df, sample_ids, maf_threshold, interaction_df,
            maf_threshold_interaction, logp, run_eigenmt, verbose, start_time,
            covariates_df
        )
        if write_stats:
            _write_chromosome_results(
                chr_res, chrom, variant_df, prefix, output_dir,
                interaction_df, logp, paired_covariate_df, mapping_state
            )
        if top_hits:
            best_assoc.extend(top_hits)

    if interaction_df is not None and best_assoc:
        return _summarize_top_associations(
            best_assoc, interaction_df, logp, run_eigenmt,
            write_top, output_dir, prefix, mapping_state, group_s
        )

    mapping_state['logger'].write('done.')


def map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
            haplotype_df=None, loci_df=None, covariates_df=None,
            group_s=None, paired_covariate_df=None, maf_threshold=0,
            beta_approx=True, nperm=10_000, window=1_000_000,
            random_tiebreak=False, logger=None, seed=None, logp=False,
            verbose=True, warn_monomorphic=True):
    """Run cis-QTL mapping with permutations (empirical p-values)."""
    # Setup and logging
    msg = "cis-QTL mapping: empirical p-values for phenotypes"
    mapping_state = _setup_mapping_inputs(
        phenotype_df, covariates_df, paired_covariate_df, phenotype_pos_df,
        maf_threshold, group_s=group_s, logger=logger, msg=msg, verbose=verbose
    )
    phenotype_df = mapping_state["phenotype_df"]
    phenotype_pos_df = mapping_state["phenotype_pos_df"]
    sample_ids = phenotype_df.columns.tolist()
    group_s = mapping_state["group_s"]
    residualizer = mapping_state["residualizer"]
    dof = mapping_state["dof"]
    device = mapping_state["device"]

    _log_mapping_context(
        genotype_df, haplotype_df, group_s, window,
        random_tiebreak, mapping_state["logger"]
    )

    # Permutation setup
    permutation_ix_t = _make_permutation_index(
        n_samples=phenotype_df.shape[1],
        nperm=nperm,
        seed=seed,
        device=device,
        logger=mapping_state["logger"]
    )

    # Input generator
    igc = local.InputGeneratorCis(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        loci_df if loci_df is not None else pd.DataFrame(index=[]),
        haplotype_df if haplotype_df is not None else pd.DataFrame(index=[]),
        group_s=group_s, window=window
    )
    if igc.n_phenotypes == 0:
        raise ValueError("No valid phenotypes found.")

    # Main loop
    res_df = []
    start_time = time.time()
    mapping_state["logger"].write("  * computing permutations")

    genotype_ix = np.array([genotype_df.columns.get_loc(s) for s in sample_ids])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    for row in igc.generate_data(verbose=verbose):
        if group_s is None:
            res_s = _process_permutation_window(
                row, genotype_ix_t, variant_df, igc, mapping_state,
                permutation_ix_t, dof, covariates_df, paired_covariate_df,
                maf_threshold, beta_approx, random_tiebreak, logp, warn_monomorphic
            )
        else:
            res_s = _process_group_permutation_window(
                row, genotype_ix_t, variant_df, igc, mapping_state,
                permutation_ix_t, dof, covariates_df, paired_covariate_df,
                maf_threshold, beta_approx, random_tiebreak, logp, warn_monomorphic
            )
        if res_s is not None:
            res_df.append(res_s)

    # Finalize
    res_df = pd.concat(res_df, axis=1, sort=False).T
    res_df.index.name = "phenotype_id"
    mapping_state["logger"].write(
        f"  Time elapsed: {(time.time() - start_time)/60:.2f} min"
    )
    mapping_state["logger"].write("done.")
    return res_df.astype(output_dtype_dict).infer_objects()


def map_independent(genotype_df, variant_df, cis_df, phenotype_df, phenotype_pos_df,
                    haplotype_df=None, loci_df=None,  covariates_df=None,
                    group_s=None, maf_threshold=0, fdr=0.05, fdr_col='qval',
                    nperm=10_000, window=1_000_000, missing=-9,
                    random_tiebreak=False, logger=None, seed=None, logp=False, verbose=True):
    """
    Run independent cis-QTL mapping (forward-backward regression)

    cis_df: output from map_cis, annotated with q-values (calculate_qvalues)
    """
    paired_covariate_df = None
    msg = 'cis-QTL mapping: conditionally independent variants'

    # Setup and logging
    mapping_state = _setup_mapping_inputs(
        phenotype_df, covariates_df, paired_covariate_df, phenotype_pos_df,
        maf_threshold, cis_df=cis_df, fdr=fdr, fdr_col=fdr_col, logger=logger,
        group_s=group_s, conditional=True, msg=msg, verbose=verbose
    )
    phenotype_df = mapping_state["phenotype_df"]
    phenotype_pos_df = mapping_state["phenotype_pos_df"]
    group_s = mapping_state["group_s"]
    device = mapping_state["device"]
    signif_df = mapping_state["signif_df"]
    signif_threshold = signif_df['pval_beta'].max()

    _log_mapping_context(
        genotype_df, haplotype_df, group_s, window,
        random_tiebreak, mapping_state["logger"]
    )

    # Permutations
    permutation_ix_t = _make_permutation_index(
        n_samples=phenotype_df.shape[1],
        nperm=nperm,
        seed=seed,
        device=device,
        logger=mapping_state["logger"]
    )

    # Index lookup for forward/backward passes
    ix_dict = {i:k for k, i in enumerate(genotype_df.index)}

    # Input generator
    igc = local.InputGeneratorCis(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        loci_df if loci_df is not None else pd.DataFrame(index=[]),
        haplotype_df if haplotype_df is not None else pd.DataFrame(index=[]),
        group_s=group_s, window=window
    )
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')

    # Main loop
    res_df = []
    start_time = time.time()

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    logger.write('  * computing independent QTLs')

    for row in igc.generate_data(verbose=verbose):
        if group_s is None:
            res_s = _process_independent_window(
                row, genotype_ix_t, variant_df, genotype_df, ix_dict, igc,
                mapping_state, permutation_ix_t, covariates_df, nperm,
                signif_threshold, maf_threshold, missing, random_tiebreak, logp
            )
        else:
            res_s = _process_group_independent_window(
                row, genotype_ix_t, variant_df, genotype_df, ix_dict, igc,
                mapping_state, permutation_ix_t, covariates_df, nperm,
                signif_threshold, maf_threshold, missing, random_tiebreak, logp
            )
        if res_s is not None:
            res_df.append(res_s)

    # Finalize
    res_df = pd.concat(res_df, axis=0, sort=False)
    res_df.index.name = "phenotype_id"
    logger.write(f"  Time elapsed: {(time.time()-start_time)/60:.2f} min")
    logger.write("done.")
    return res_df.reset_index().astype(output_dtype_dict)
