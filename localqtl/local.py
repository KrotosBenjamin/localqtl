"""
GPU-enabled utilities to incorporate local ancestry (RFMix) into tensorQTL-style
cis mapping. Provides:
  - RFMixReader: aligns RFMix local-ancestry to genotype variant order (lazy via dask/zarr)
  - get_cis_ranges: computes per-phenotype cis windows for BOTH variants and haplotypes
  - InputGeneratorCis: background-prefetched batch generator that yields
      phenotype, variants slice, haplotypes slice, their index ranges, and IDs

Notes
-----
- Designed for large-scale GPU eQTL with CuPy/cuDF where possible.
- Avoids materialization; uses dask-backed arrays and cuDF slicing.
- Compatible with original tensorQTL patterns while adding local ancestry.

Author: Kynon J Benjamin (refactor by assistant per user request)
"""
from __future__ import annotations

# ----------------------------
# Imports
# ----------------------------
import bisect, sys
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Tuple, Union

import dask.array as da
from dask.array import from_array, from_zarr

import cupy as cp
from cupy import int8, int32

import cudf
from cudf import DataFrame as cuDF, Series as cuSeries

from tensorqtl.genotypeio import background
from rfmix_reader import read_rfmix, interpolate_array


ArrayLike = Union[np.ndarray, cp.ndarray, da.core.Array]

# ----------------------------
# RFMixReader (refined)
# ----------------------------
class RFMixReader:
    """Read and align RFMix local ancestry to variant grid.

    Parameters
    ----------
    prefix_path : str
        Directory containing RFMix per-chrom outputs and fb.tsv.
    variant_df : pd.DataFrame
        DataFrame with columns ['chrom', 'pos'] in the SAME order as the
        genotype matrix (variants x samples).
    select_samples : list[str], optional
        Subset of sample IDs to keep (order preserved).
    exclude_chrs : list[str], optional
        Chromosomes to exclude from imputed matrices.
    binary_dir_name : str
        Directory name with prebuilt binary files (default: "binary_files").
    verbose : bool
    dtype : cupy dtype

    Attributes
    ----------
    loci : cuDF
        Imputed loci aligned to variants (columns: ['chrom','pos','i','hap']).
    admix : dask.array
        Dask array (loci x (n_samples * n_pops)) or shaped after selection.
    rf_q : cuDF
        Sample metadata table from RFMix (contains 'sample_id', 'chrom').
    sample_ids : list[str]
    n_pops : int
    hap_df : pd.DataFrame
        Mapping hap_id -> (chrom, pos, index) for fast lookups.
    hap_dfs : dict[str, pd.DataFrame]
        Per-chrom position/index tables for windowing.
    """

    def __init__(
        self, prefix_path: str, variant_df: pd.DataFrame,
        select_samples: Optional[List[str]] = None,
        exclude_chrs: Optional[List[str]] = None,
        binary_dir_name: str = "binary_files",
        verbose: bool = True, dtype=int8,
    ):
        self.zarr_dir = f"{prefix_path}"
        bin_dir = f"{prefix_path}/{binary_dir_name}"

        loci, self.rf_q, admix = read_rfmix(prefix_path, binary_dir=bin_dir,
                                            verbose=verbose)
        loci = loci.rename(columns={"chromosome": "chrom",
                                    "physical_position": "pos"})

        # Ensure unique variant positions for merge/alignment
        variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"],
                                                keep="first").copy()

        # Align variant grid with available loci (allow imputation for missing)
        # Merge retains full variant grid; keep indices for direct selection
        variant_loci = (
            variant_df.merge(_to_pandas(loci), on=["chrom", "pos"], how="outer",
                             indicator=True)
            .loc[:, ["chrom", "pos", "i", "_merge"]]
        )

        # Drive imputation to build a complete ancestry grid aligned to variants
        _ = interpolate_array(variant_loci, admix, self.zarr_dir)
        daz = from_zarr(f"{self.zarr_dir}/local-ancestry.zarr")  # (variants_aligned x samples*pops)

        # Indices present in original loci (not right_only) to map back
        idx_arr = from_array(
            variant_loci[~(variant_loci["_merge"] == "right_only")].index.to_numpy()
        )
        self.admix = daz[idx_arr]  # dask view, lazy

        # Build cuDF mask to select aligned rows; drop helper cols
        mask = cudf.Series(False, index=variant_loci.index)
        mask.loc[idx_arr.compute()] = True
        variant_loci = cudf.from_pandas(variant_loci)
        self.loci = (
            variant_loci[mask]
            .drop(["i", "_merge"], axis=1)
            .reset_index(drop=True)
        )
        self.loci["i"] = cudf.Series(range(len(self.loci)))
        self.loci["hap"] = self.loci["chrom"].astype(str) + "_" + self.loci["pos"].astype(str)

        # Use all samples by default; allow subsetting
        self.sample_ids = _get_sample_ids(self.rf_q)
        if select_samples is not None:
            ix = [self.sample_ids.index(i) for i in select_samples]
            self.admix = self.admix[:, ix]
            self.rf_q = self.rf_q.loc[ix].reset_index(drop=True)
            self.sample_ids = _get_sample_ids(self.rf_q)

        # Exclude chromosomes if requested
        if exclude_chrs is not None:
            mask = ~self.loci["chrom"].isin(exclude_chrs)
            self.admix = self.admix[mask.values, :]
            self.loci = self.loci[mask].copy().reset_index(drop=True)
            self.loci["i"] = self.loci.index

        # Populations inferred from columns
        n_samples = len(self.sample_ids)
        n_cols = self.admix.shape[1]
        if (n_cols % n_samples) != 0:
            raise ValueError("Admix columns not multiple of n_samples; cannot infer n_pops.")
        self.n_pops = n_cols // n_samples

        # Haplotype lookup frames (CPU pandas for bisect speed)
        hap_df = self.loci.to_pandas().set_index("hap")[['chrom', 'pos']]
        hap_df['index'] = np.arange(hap_df.shape[0])
        self.hap_df = hap_df
        self.hap_dfs = {c: g[["pos", "index"]].sort_values("pos").reset_index(drop=True)
                        for c, g in hap_df.reset_index().groupby('chrom', sort=False)}


# ----------------------------
# Helpers functions
# ----------------------------
def _to_pandas(df: Union[cuDF, pd.DataFrame]) -> pd.DataFrame:
    return df.to_pandas() if isinstance(df, cuDF) else df


def _get_sample_ids(df: Union[cuDF, pd.DataFrame]) -> List[str]:
    if isinstance(df, (cuDF, cuSeries)):
        return df["sample_id"].to_arrow().to_pylist()
    return df["sample_id"].tolist()


# -------------------------------------------------
# cis-window computation for variants + haplotypes
# -------------------------------------------------
def get_cis_ranges(
    phenotype_pos_df: pd.DataFrame,
    chr_variant_dfs: Dict[str, pd.DataFrame],
    chr_haplotype_dfs: Dict[str, pd.DataFrame],
    window: int, require_both: bool = True, verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, Tuple[int, int]]], List[str]]:
    """Compute per-phenotype cis index ranges for variants and haplotypes.

    Returns
    -------
    cis_ranges : dict
        phenotype_id -> {"variants": (lb, ub), "haplotypes": (lb, ub)} (inclusive ranges)
    drop_ids : list[str]
        Phenotypes without any eligible window (based on `require_both`).
    """
    # Normalize phenotype_pos_df to have ['chr','start','end']
    if 'pos' in phenotype_pos_df.columns:
        pp = phenotype_pos_df.rename(columns={'pos': 'start'}).copy()
        pp['end'] = pp['start']
    else:
        pp = phenotype_pos_df.copy()

    # Ensure dict-of-records for speed
    phenotype_pos_dict = pp.to_dict(orient='index')

    cis_ranges: Dict[str, Dict[str, Tuple[int, int]]] = {}
    drop_ids: List[str] = []

    # Pre-extract numpy arrays for bisect per chromosome
    var_pos = {c: df['pos'].to_numpy() for c, df in chr_variant_dfs.items()}
    var_idx = {c: df['index'].to_numpy() for c, df in chr_variant_dfs.items()}
    hap_pos = {c: df['pos'].to_numpy() for c, df in chr_haplotype_dfs.items()}
    hap_idx = {c: df['index'].to_numpy() for c, df in chr_haplotype_dfs.items()}

    ids = list(phenotype_pos_df.index)
    n = len(ids)
    for k, pid in enumerate(ids, 1):
        if verbose and (k % 1000 == 0 or k == n):
            print(f"\r  * checking phenotypes: {k}/{n}", end='' if k != n else None)
        pos = phenotype_pos_dict[pid]
        chrom = pos['chr']

        # Variants
        if chrom in var_pos:
            lb = bisect.bisect_left(var_pos[chrom], pos['start'] - window)
            ub = bisect.bisect_right(var_pos[chrom], pos['end'] + window) - 1
            variant_r = (var_idx[chrom][lb], var_idx[chrom][ub]) if lb <= ub else None
        else:
            variant_r = None

        # Haplotypes
        if chrom in hap_pos:
            lb = bisect.bisect_left(hap_pos[chrom], pos['start'] - window)
            ub = bisect.bisect_right(hap_pos[chrom], pos['end'] + window) - 1
            haplotype_r = (hap_idx[chrom][lb], hap_idx[chrom][ub]) if lb <= ub else None
        else:
            haplotype_r = None

        ok = (variant_r is not None) and (haplotype_r is not None) if require_both \
             else (variant_r is not None) or (haplotype_r is not None)
        if ok:
            cis_ranges[pid] = {"variants": variant_r, "haplotypes": haplotype_r}
        else:
            drop_ids.append(pid)

    return cis_ranges, drop_ids


# ----------------------------
# Input generator (refactored)
# ----------------------------
class InputGeneratorCis:
    """Input generator for cis mapping (variants + local ancestry haplotypes).

    Inputs
    ------
    genotype_df : (variants x samples) DataFrame (pd or cuDF)
    variant_df  : DataFrame mapping variant index to ['chrom','pos'] (sorted by genotype row order)
    phenotype_df: (phenotypes x samples) DataFrame
    phenotype_pos_df: DataFrame with ['chr','pos'] or ['chr','start','end'] indexed by phenotype_id
    loci_df     : (haplotypes x samples) DataFrame aligned to hap_df order
    hap_df      : DataFrame with index hap_id and columns ['chrom','pos'] in row order matching loci_df
    group_s     : optional pd.Series mapping phenotype_id -> group_id
    window      : cis window size

    Generates (ungrouped)
    --------------------
    phenotype (1D), variants (2D slice), variants_index (1D),
    haplotypes (2D slice), haplotypes_index (1D), phenotype_id

    Notes
    -----
    - Uses background prefetch for overlap with compute.
    - Returns GPU arrays (CuPy) if as_cupy=True, else returns DataFrames.
    """

    def __init__(
        self,
        genotype_df: Union[pd.DataFrame, cuDF],
        variant_df: pd.DataFrame,
        phenotype_df: Union[pd.DataFrame, cuDF],
        phenotype_pos_df: pd.DataFrame,
        loci_df: Union[pd.DataFrame, cuDF],
        hap_df: pd.DataFrame,
        group_s: Optional[pd.Series] = None,
        window: int = 1_000_000,
        require_both: bool = True,
    ):
        # Store
        self.genotype_df = genotype_df
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(self.variant_df.shape[0])

        self.loci_df = loci_df
        self.hap_df = hap_df.copy()
        self.hap_df['index'] = np.arange(self.hap_df.shape[0])

        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df.copy()

        self.group_s = group_s
        self.window = window
        self.require_both = require_both

        # Validate & filter
        self._validate_data()
        self._filter_phenotypes_by_genotypes()
        self._filter_phenotypes_by_haplotypes()
        self._drop_constant_phenotypes()
        self._calculate_cis_ranges()

    # ----------------------------
    # Validation & filtering
    # ----------------------------
    def _validate_data(self):
        # Index alignment
        assert (self.genotype_df.index == self.variant_df.index).all(), \
            "Genotype and variant DataFrames must share the same index order."
        assert (self.hap_df.index == self.loci_df.index).all(), \
            "Haplotype (hap_df) and loci_df must share the same index order."
        # Phenotype index uniqueness
        ph_index = self._to_pandas(self.phenotype_df).index
        assert (ph_index == pd.Index(ph_index).unique()).all(), \
            "Phenotype DataFrame index must be unique."

    def _filter_phenotypes_by_genotypes(self):
        variant_chrs = pd.Index(self.variant_df['chrom'].unique())
        phenotype_chrs = pd.Index(self.phenotype_pos_df['chr'].unique())
        keep_chrs = phenotype_chrs.intersection(variant_chrs)
        m = self.phenotype_pos_df['chr'].isin(keep_chrs)
        drop_n = int((~m).sum())
        if drop_n:
            print(f"    ** dropping {drop_n} phenotypes on chrs. without genotypes")
        self.phenotype_df = self._loc_idx(self.phenotype_df, m)
        self.phenotype_pos_df = self.phenotype_pos_df.loc[m]

    def _filter_phenotypes_by_haplotypes(self):
        hap_chrs = pd.Index(self.hap_df['chrom'].unique())
        phenotype_chrs = pd.Index(self.phenotype_pos_df['chr'].unique())
        keep_chrs = phenotype_chrs.intersection(hap_chrs)
        m = self.phenotype_pos_df['chr'].isin(keep_chrs)
        drop_n = int((~m).sum())
        if drop_n:
            print(f"    ** dropping {drop_n} phenotypes on chrs. without haplotypes")
        self.phenotype_df = self._loc_idx(self.phenotype_df, m)
        self.phenotype_pos_df = self.phenotype_pos_df.loc[m]

    def _drop_constant_phenotypes(self):
        P = self._to_pandas(self.phenotype_df).values
        # constant across samples
        m = np.all(P == P[:, [0]], axis=1)
        drop_n = int(m.sum())
        if drop_n:
            print(f"    ** dropping {drop_n} constant phenotypes")
            self.phenotype_df = self._loc_idx(self.phenotype_df, ~m)
            self.phenotype_pos_df = self.phenotype_pos_df.loc[~m]
        if len(self._to_pandas(self.phenotype_df)) == 0:
            raise ValueError("No phenotypes remain after filters.")

    def _calculate_cis_ranges(self):
        # Build per-chrom position/index tables (sorted)
        self.chr_variant_dfs = {c: g[['pos', 'index']].sort_values('pos').reset_index(drop=True)
                                for c, g in self.variant_df.groupby('chrom', sort=False)}
        self.chr_haplotype_dfs = {c: g[['pos', 'index']].sort_values('pos').reset_index(drop=True)
                                  for c, g in self.hap_df.groupby('chrom', sort=False)}

        self.cis_ranges, drop_ids = get_cis_ranges(
            self.phenotype_pos_df,
            self.chr_variant_dfs,
            self.chr_haplotype_dfs,
            self.window,
            require_both=self.require_both,
            verbose=True,
        )
        if drop_ids:
            print(f"    ** dropping {len(drop_ids)} phenotypes without required windows")
            self.phenotype_df = self._drop_by_ids(self.phenotype_df, drop_ids)
            self.phenotype_pos_df = self.phenotype_pos_df.drop(drop_ids)

        # Cache counts
        self.n_phenotypes = self._to_pandas(self.phenotype_df).shape[0]
        if self.group_s is not None:
            self.group_s = self.group_s.loc[self.phenotype_pos_df.index].copy()
            self.n_groups = int(self.group_s.unique().shape[0])

        # Phenotype start/end dicts
        if 'pos' in self.phenotype_pos_df.columns:
            self.phenotype_start = self.phenotype_pos_df['pos'].to_dict()
            self.phenotype_end = self.phenotype_start
        else:
            self.phenotype_start = self.phenotype_pos_df['start'].to_dict()
            self.phenotype_end = self.phenotype_pos_df['end'].to_dict()

    # ----------------------------
    # Generation
    # ----------------------------
    @background(max_prefetch=6)
    def generate_data(
        self, chrom: Optional[str] = None,
        verbose: bool = False, as_cupy: bool = True,
    ):
        """Yield batches for cis mapping.

        Yields
        ------
        phenotype: 1D array (samples,)
        variants:  2D array (n_variants_in_window x samples)
        v_index:   1D array of variant row indices (global)
        haplotypes:2D array (n_haps_in_window x samples)
        h_index:   1D array of haplotype row indices (global)
        phenotype_id: str or list[str] if grouped
        [group_id]: optional, when grouped
        """
        if chrom is None:
            phenotype_ids = list(self.phenotype_pos_df.index)
            chr_offset = 0
        else:
            phenotype_ids = list(self.phenotype_pos_df[self.phenotype_pos_df['chr'] == chrom].index)
            # compute offset for cosmetic progress (optional)
            offset_dict = {c: i for i, c in enumerate(self.phenotype_pos_df['chr'].drop_duplicates())}
            chr_offset = int(offset_dict.get(chrom, 0))

        index_of = {pid: i for i, pid in enumerate(self.phenotype_df.index)}

        if self.group_s is None:
            for k, pid in enumerate(phenotype_ids, chr_offset + 1):
                if verbose:
                    _print_progress(k, self.n_phenotypes, 'phenotype')

                p = _row(self.phenotype_df, index_of[pid], as_cupy=as_cupy).ravel()
                r = self.cis_ranges[pid]

                # Variant slice
                v_lb, v_ub = r['variants'] if r['variants'] is not None else (None, None)
                if v_lb is not None:
                    G = _row_slice(self.genotype_df, v_lb, v_ub + 1, as_cupy=as_cupy)
                    G_idx = np.arange(v_lb, v_ub + 1)
                else:
                    G = None
                    G_idx = np.arange(0, 0, dtype=int)

                # Haplotype slice
                h_lb, h_ub = r['haplotypes'] if r['haplotypes'] is not None else (None, None)
                if h_lb is not None:
                    H = _row_slice(self.loci_df, h_lb, h_ub + 1, as_cupy=as_cupy)
                    H_idx = np.arange(h_lb, h_ub + 1)
                else:
                    H = None
                    H_idx = np.arange(0, 0, dtype=int)

                yield p, G, G_idx, H, H_idx, pid
        else:
            # Grouped mode: all phenotypes in group must share ranges or we take union
            grouped = self.group_s.loc[phenotype_ids].groupby(self.group_s, sort=False)
            for k, (group_id, g) in enumerate(grouped, chr_offset + 1):
                if verbose:
                    _print_progress(k, self.n_groups, 'phenotype group')
                ids = list(g.index)
                idxs = [index_of[i] for i in ids]
                p = _rows(self.phenotype_df, idxs, as_cupy=as_cupy)

                # Validate identical ranges; if not, take union
                ranges = [self.cis_ranges[i] for i in ids]
                v_lbs = [r['variants'][0] for r in ranges if r['variants'] is not None]
                v_ubs = [r['variants'][1] for r in ranges if r['variants'] is not None]
                h_lbs = [r['haplotypes'][0] for r in ranges if r['haplotypes'] is not None]
                h_ubs = [r['haplotypes'][1] for r in ranges if r['haplotypes'] is not None]

                v_lb, v_ub = (min(v_lbs), max(v_ubs)) if len(v_lbs) else (None, None)
                h_lb, h_ub = (min(h_lbs), max(h_ubs)) if len(h_lbs) else (None, None)

                G = _row_slice(self.genotype_df, v_lb, (v_ub + 1) if v_ub is not None else None, as_cupy=as_cupy) if v_lb is not None else None
                H = _row_slice(self.loci_df, h_lb, (h_ub + 1) if h_ub is not None else None, as_cupy=as_cupy) if h_lb is not None else None
                G_idx = np.arange(v_lb, v_ub + 1) if v_lb is not None else np.arange(0, 0, dtype=int)
                H_idx = np.arange(h_lb, h_ub + 1) if h_lb is not None else np.arange(0, 0, dtype=int)

                yield p, G, G_idx, H, H_idx, ids, group_id

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _to_pandas(df: Union[pd.DataFrame, cuDF]) -> pd.DataFrame:
        return df.to_pandas() if isinstance(df, cuDF) else df

    @staticmethod
    def _loc_idx(df: Union[pd.DataFrame, cuDF], mask: Union[np.ndarray, pd.Series]) -> Union[pd.DataFrame, cuDF]:
        if isinstance(df, cuDF):
            return df.loc[cudf.from_pandas(pd.Series(mask)).to_pandas().to_numpy()]
        return df.loc[mask]

    @staticmethod
    def _drop_by_ids(df: Union[pd.DataFrame, cuDF], ids: List[str]) -> Union[pd.DataFrame, cuDF]:
        if isinstance(df, cuDF):
            return df.drop(ids, errors='ignore')
        return df.drop(index=ids, errors='ignore')


# ------------------------------------------------------
# Low-level row getters (avoid .values materialization)
# ------------------------------------------------------
def _row(df: Union[pd.DataFrame, cuDF], i: int, as_cupy: bool = True) -> ArrayLike:
    if isinstance(df, cuDF):
        arr = df.iloc[i]
        return arr.to_cupy() if as_cupy else arr
    # pandas
    arr = df.iloc[i].to_numpy(copy=False)
    return cp.asarray(arr) if as_cupy else arr


def _rows(df: Union[pd.DataFrame, cuDF], idxs: List[int], as_cupy: bool = True) -> ArrayLike:
    if isinstance(df, cuDF):
        arr = df.iloc[idxs]
        return arr.to_cupy() if as_cupy else arr
    arr = df.iloc[idxs].to_numpy(copy=False)
    return cp.asarray(arr) if as_cupy else arr


def _row_slice(
    df: Union[pd.DataFrame, cuDF], lb: Optional[int], ub: Optional[int], as_cupy: bool = True
) -> Optional[ArrayLike]:
    if lb is None:
        return None
    if isinstance(df, cuDF):
        view = df.iloc[lb:ub]
        return view.to_cupy() if as_cupy else view
    view = df.iloc[lb:ub].to_numpy(copy=False)
    return cp.asarray(view) if as_cupy else view


# ----------------------------
# Simple progress printer
# ----------------------------
def _print_progress(k: int, n: int, entity: str) -> None:
    msg = f"\r    processing {entity} {k}/{n}"
    if k == n:
        msg += "\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
