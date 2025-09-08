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

Author: Kynon J Benjamin
"""
from __future__ import annotations

# ----------------------------
# Imports
# ----------------------------
import zarr
import bisect, sys
import numpy as np
import pandas as pd
import dask.array as da
from os.path import exists
from typing import Dict, List, Optional, Tuple, Union

from genotypeio import background
from rfmix_reader import read_rfmix, interpolate_array

import cudf
import cupy as cp
from cudf import DataFrame as cuDF

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
    select_samples : list[str], optional
        Subset of sample IDs to keep (order preserved).
    exclude_chrs : list[str], optional
        Chromosomes to exclude from imputed matrices.
    binary_path : str
        Path with prebuilt binary files (default: "./binary_files").
    verbose : bool
    dtype : numpy dtype

    Attributes
    ----------
    loci : cuDF
        Imputed loci aligned to variants (columns: ['chrom','pos','i','hap']).
    admix : dask.array
        Dask array with shape (loci, samples, ancestries)
    g_anc : cuDF or pd.DataFrame
        Sample metadata table from RFMix (contains 'sample_id', 'chrom').
    sample_ids : list[str]
    n_pops : int
    loci_df : pd.DataFrame
        Ancestry dosage aligned to hap_df.
    haplotypes : dask.array
        Haplotype-level ancestry matrix (variants x samples [x ancestries]).
    """

    def __init__(
        self, prefix_path: str, #variant_df: pd.DataFrame,
        select_samples: Optional[List[str]] = None,
        exclude_chrs: Optional[List[str]] = None,
        binary_path: str = "./binary_files",
        verbose: bool = True, dtype=np.int8
    ):
        # self.zarr_dir = f"{prefix_path}"
        bin_dir = f"{binary_path}"

        self.loci, self.g_anc, self.admix = read_rfmix(prefix_path,
                                                       binary_dir=bin_dir,
                                                       verbose=verbose)
        if self.admix.ndim != 3:
            n_vars, total = self.admix.shape
            n_pops = total // len(self.g_anc.sample_id.unique())
            n_samp = total // n_pops
            self.admix = self.admix.reshape(n_vars, n_samp, n_pops)

        # Guard unknown shapes
        if any(dim is None for dim in self.admix.shape):
            raise ValueError(
                "Ancestry array has unknown dimensions; expected (variants, samples, ancestries)."
            )

        # Build loci table
        self.loci = self.loci.rename(columns={"chromosome": "chrom",
                                              "physical_position": "pos"})
        self.loci["i"] = cudf.Series(range(len(self.loci)))
        self.loci["hap"] = self.loci["chrom"].astype(str) + "_" + self.loci["pos"].astype(str)

        # Subset samples
        self.sample_ids = _get_sample_ids(self.g_anc)
        if select_samples is not None:
            ix = [self.sample_ids.index(i) for i in select_samples]
            self.admix = self.admix[:, ix, :]
            if isinstance(self.g_anc, cuDF):
                self.g_anc = self.g_anc.loc[ix].reset_index(drop=True)
            else:
                self.g_anc = self.g_anc.iloc[ix].reset_index(drop=True)
            self.sample_ids = _get_sample_ids(self.g_anc)

        # Exclude chromosomes if requested
        if exclude_chrs is not None and len(exclude_chrs) > 0:
            mask_pd = ~self.loci.to_pandas()["chrom"].isin(exclude_chrs).values
            self.admix = self.admix[mask_pd, :, :]
            keep_idx = np.nonzero(mask_pd)[0]
            self.loci = self.loci[keep_idx].reset_index(drop=True)
            self.loci["i"] = self.loci.index

        # Dimensions
        self.n_samples = int(self.admix.shape[1])
        self.n_pops = int(self.admix.shape[2])

        # Build hap tables
        if self.n_pops == 2:
            A0 = self.admix[:, :, [0]]
            loci_ids = (self.loci["chrom"].astype(str) + "_" + self.loci["pos"].astype(str) + "_A0")
            loci_df = self.loci.to_pandas()[["chrom", "pos"]].copy()
            loci_df["ancestry"] = 0
            loci_df["hap"] = _to_pandas(loci_ids)
            loci_df["index"] = np.arange(loci_df.shape[0])
            self.loci_df = loci_df.set_index("hap")
            self.loci_dfs = {c: g[["pos", "index"]].sort_values("pos").reset_index(drop=True)
                            for c, g in self.loci_df.reset_index().groupby("chrom", sort=False)}
            self.haplotypes = A0
        else: # >2 ancestries
            loci_dfs = []
            for anc in range(self.n_pops):
                loci_df_anc = self.loci.to_pandas()[["chrom", "pos"]].copy()
                loci_df_anc["ancestry"] = anc
                loci_df_anc["hap"] = (
                    loci_df_anc["chrom"].astype(str) + "_" + loci_df_anc["pos"].astype(str) + f"_A{anc}"
                )
                # Global index along flattened (variants*ancestries) axis
                loci_df_anc["index"] = np.arange(loci_df_anc.shape[0]) + anc * self.loci.shape[0]
                loci_dfs.append(loci_df_anc)

            self.loci_df = pd.concat(loci_dfs).set_index("hap")
            self.loci_dfs = {c: g[["pos", "index", "ancestry"]].sort_values("pos").reset_index(drop=True)
                            for c, g in self.loci_df.reset_index().groupby("chrom", sort=False)}
            self.haplotypes = self.admix  # dask array

    def load_haplotypes(self):
        """Force-load haplotype ancestry into memory as NumPy array."""
        return self.haplotypes.compute()

# -------------------------------------------------
# cis-window computation for variants + haplotypes
# -------------------------------------------------
def get_cis_ranges(
    phenotype_pos_df: pd.DataFrame,
    chr_variant_dfs: Dict[str, pd.DataFrame],
    window: int, verbose: bool = True):
    """Compute per-phenotype cis index ranges for variants.

    Returns
    -------
    cis_ranges : dict
        phenotype_id -> {"variants": (lb, ub)
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

    drop_ids = []
    cis_ranges = {}
    ids = list(phenotype_pos_df.index)
    n = len(ids)
    for k, pid in enumerate(ids, 1):
        if verbose and (k % 1000 == 0 or k == n):
            print(f"\r  * checking phenotypes: {k}/{n}", end='' if k != n else None)

        pos = phenotype_pos_dict[pid]
        chrom = pos['chr']

        # Variants
        variant_r = None
        if chrom in chr_variant_dfs:
            pos_array = chr_variant_dfs[chrom]['pos'].values
            lb = bisect.bisect_left(pos_array, pos['start'] - window)
            ub = bisect.bisect_right(pos_array, pos['end'] + window)
            if lb < ub:
                variant_r = (lb, ub - 1)

        if variant_r is not None:
            cis_ranges[pid] = variant_r
        else:
            drop_ids.append(pid)

    return cis_ranges, drop_ids


# -------------------------------
# Input generator for haplotypes
# -------------------------------
class InputGeneratorCis:
    """Input generator for cis mapping (variants + local ancestry haplotypes).

    Inputs
    ------
    genotype_df : (variants x samples) DataFrame
    variant_df  : DataFrame mapping variant index to ['chrom','pos'] (sorted by genotype row order)
    phenotype_df: (phenotypes x samples) DataFrame
    phenotype_pos_df: DataFrame with ['chr','pos'] or ['chr','start','end'] indexed by phenotype_id
    haplotypes  : Dask array or NumPy array (haplotypes x samples x ancestries)
    loci_df     : DataFrame with index hap_id and columns ['chrom','pos'] in row order matching haplotypes
    group_s     : optional pd.Series mapping phenotype_id -> group_id
    window      : cis window size
    on_the_fly_impute : optional bool to impute haplotypes (default: True). If FLARE set to False.

    Generates (ungrouped)
    --------------------
    phenotype (1D), variants (2D slice), variants_index (1D),
    haplotypes (2D slice), haplotypes_index (1D), phenotype_id
    """

    def __init__(
        self,
        genotype_df: pd.DataFrame,
        variant_df: pd.DataFrame,
        phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame,
        haplotypes: Union[pd.DataFrame, cuDF, da.Array, np.ndarray],
        loci_df: Union[pd.DataFrame, cuDF],
        group_s: Optional[pd.Series] = None,
        window: int = 1_000_000,
        require_both: bool = True,
        on_the_fly_impute: bool = True,
    ):
        # Store
        self.genotype_df = genotype_df
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(self.variant_df.shape[0])

        self.loci_df = loci_df.copy()
        self.loci_df['index'] = np.arange(self.loci_df.shape[0])
        self.haplotypes = haplotypes  # Keep Zarr array
        self.on_the_fly_impute = on_the_fly_impute

        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df.copy()

        self.n_samples = self._to_pandas(self.phenotype_df).shape[1]

        self.group_s = group_s
        self.window = window
        self.require_both = require_both

        # Validate & filter
        self._validate_data()
        self._filter_phenotypes_by_genotypes()
        self._drop_constant_phenotypes()
        self._calculate_cis_ranges()

    # ----------------------------
    # Validation & filtering
    # ----------------------------
    def _validate_data(self):
        # Index alignment
        assert (self.genotype_df.index == self.variant_df.index).all(), \
            "Genotype and variant DataFrames must share the same index order."
        # Haplotype data
        if isinstance(self.haplotypes, (pd.DataFrame, cuDF)):
            assert self.haplotypes.shape[0] == len(self.loci_df), \
                "Haplotypes rows must equal loci information length."
        elif isinstance(self.haplotypes, (da.Array, np.ndarray)):
            assert int(self.haplotypes.shape[0]) == len(self.loci_df), \
                "Haplotypes (dask) first dim must equal loci information length."
        # Phenotype index uniqueness
        ph_index = self._to_pandas(self.phenotype_df).index
        assert (ph_index == pd.Index(ph_index).unique()).all(), \
            "Phenotype DataFrame index must be unique."
        # Phenotype index alignment (important for masks)
        assert ph_index.equals(self.phenotype_pos_df.index), \
            "Phenotype DataFrame and position must have identical index order."

    def _loc_idx(self, df: Union[pd.DataFrame, cuDF], mask: Union[np.ndarray, pd.Series]
                 ) -> Union[pd.DataFrame, cuDF]:
        """Boolean row filter that supports pandas/cuDF with a numpy/pandas mask."""
        if isinstance(df, cuDF):
            mask_arr = mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)
            return df.loc[cudf.Series(mask_arr)]
        return df.loc[mask]

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
        self.chrs = list(keep_chrs)

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
        self.chr_variant_dfs = {
            c: g[['pos', 'index']].sort_values('pos').reset_index(drop=True)
            for c, g in self.variant_df.groupby('chrom', sort=False)
        }

        self.cis_ranges, drop_ids = get_cis_ranges(
            self.phenotype_pos_df,
            self.chr_variant_dfs,
            self.window,
            verbose=True,
        )

        if drop_ids:
            print(f"    ** dropping {len(drop_ids)} phenotypes without required windows")
            self.phenotype_df = self._drop_by_ids(self.phenotype_df, drop_ids)
            self.phenotype_pos_df = self.phenotype_pos_df.drop(drop_ids)

        self.cis_v_idx = {}
        for pid, r in self.cis_ranges.items():
            if r is None:
                continue
            v_lb, v_ub = r
            chrom = self.phenotype_pos_df.at[pid, 'chr']
            idx_arr = self.chr_variant_dfs[chrom]['index'].to_numpy()[v_lb : v_ub + 1]
            self.cis_v_idx[pid] = idx_arr.astype(int, copy=False)

        # Cache counts
        self.n_phenotypes = int(self._to_pandas(self.phenotype_df).shape[0])
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

    @staticmethod
    def _interpolate_block(block) -> "np.ndarray":
        """
        Interpolate missing values in a 3D haplotype block: (loci, samples, ancestries).

        Performs linear interpolation along the loci axis (axis=0) for each (sample, ancestry)
        pair independently. Supports NumPy or CuPy arrays via arr_mod.

        Parameters
        ----------
        block : arr_mod.ndarray
            Haplotype slice of shape (loci, samples, ancestries), potentially with NaNs.

        Returns
        -------
        arr_mod.ndarray
            Same shape as input, with NaNs interpolated (and rounded to integers).
        """
        # Determine arrray module
        mod = cp.get_array_module(block) if cp and isinstance(block, cp.ndarray) else np

        loci_dim, sample_dim, ancestry_dim = block.shape
        block = block.reshape(loci_dim, -1)  # Shape: (loci, samples * ancestries)
        idx = mod.arange(loci_dim)

        block_imputed = block.copy()

        for s in range(sample_dim):
            col = block[:, s]
            mask = mod.isnan(col)
            if mod.any(mask):
                valid = ~mask
                if mod.any(valid):
                    # Linear interpolation and rounding
                    interpolated = mod.interp(idx[mask], idx[valid], col[valid])
                    block_imputed[mask, s] = mod.round(interpolated).astype(int)

        return block_imputed.reshape(loci_dim, sample_dim, ancestry_dim)

    # ----------------------------
    # Dask-aware row slicers
    # ----------------------------
    @staticmethod
    def _slice_rows(arr, lb: Optional[int], ub: Optional[int]):
        """Row slice from DF/cuDF/Zarr/NumPy."""
        if lb is None or ub is None or lb < 0 or ub <= lb:
            return None
        if isinstance(arr, (pd.DataFrame, cuDF)):
            return arr.iloc[lb:ub].to_numpy()
        if isinstance(arr, (zarr.Array, np.ndarray)):
            return np.asarray(arr[lb:ub])
        return TypeError(f"Unsupported haplotype type: {type(arr)}")

    @staticmethod
    def _row(arr, i: int):
        if isinstance(arr, (pd.DataFrame, cuDF)):
            return arr.iloc[i].to_numpy()
        if isinstance(arr, (zarr.Array, np.ndarray)):
            return np.asarray(arr[i])
        raise TypeError(f"Unsupported haplotype type: {type(arr)}")

    @staticmethod
    def _rows(arr, idxs: List[int]):
        if isinstance(arr, (pd.DataFrame, cuDF)):
            return arr.iloc[idxs].to_numpy()
        if isinstance(arr, (zarr.Array, np.ndarray)):
            return np.asarray(arr[idxs])
        raise TypeError(f"Unsupported haplotype type: {type(arr)}")

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _drop_by_ids(df: Union[pd.DataFrame, cuDF], ids: List[str]) -> Union[pd.DataFrame, cuDF]:
        if isinstance(df, cuDF):
            return df.drop(ids, errors='ignore')
        return df.drop(index=ids, errors='ignore')

    @staticmethod
    def _to_pandas(df: Union[pd.DataFrame, cuDF]) -> pd.DataFrame:
        return df.to_pandas() if isinstance(df, cuDF) else df

    # ----------------------------
    # Generation
    # ----------------------------
    @background(max_prefetch=6)
    def generate_data(
        self, chrom: Optional[str] = None, verbose: bool = False,
        as_cupy: bool = True, debug: bool = False,
    ):
        """
        Yield batches for cis mapping with on-the-fly haplotype imputation.

        Yields
        ------
        phenotype: 1D array (samples,)
        variants:  2D array (n_variants_in_window x samples)
        v_index:   1D array of variant row indices (global)
        haplotypes:2D array (n_haps_in_window x samples)
        phenotype_id: str or list[str] if grouped
        [group_id]: optional, when grouped
        """
        if chrom is None:
            phenotype_ids = list(self.phenotype_pos_df.index)
            chr_offset = 0
        else:
            phenotype_ids = list(self.phenotype_pos_df[self.phenotype_pos_df['chr'] == chrom].index)
            offset_dict = {c: i for i, c in enumerate(self.phenotype_pos_df['chr'].drop_duplicates())}
            chr_offset = int(offset_dict.get(chrom, 0))

        index_of = {pid: i for i, pid in enumerate(self.phenotype_df.index)}

        if self.group_s is None:
            for k, pid in enumerate(phenotype_ids, chr_offset + 1):
                if verbose:
                    _print_progress(k, self.n_phenotypes, 'phenotype')

                p = self._row(self.phenotype_df, index_of[pid]).ravel()
                v_idx = self.cis_v_idx.get(pid, None)
                if v_idx is None or v_idx.size == 0:
                    continue

                G = self._rows(self.genotype_df, v_idx)
                G_idx = v_idx

                if isinstance(self.haplotypes, (da.Array, zarr.Array, np.ndarray, pd.DataFrame, cuDF)):
                    H_slice = self.haplotypes[v_idx, :, :]      # (v, samples, k) or (v, samples) if k==1
                else:
                    raise TypeError(f"Unsupported haplotype type: {type(self.haplotypes)}")

                if self.on_the_fly_impute:
                    H_block = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)
                    H = self._interpolate_block(H_block)
                else:
                    H = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)

                # Optional squeeze when k==1 for downstream (v, s) shape
                if H is not None and H.ndim == 3 and H.shape[2] == 1:
                    H = H.reshape(H.shape[0], H.shape[1])

                yield p, G, G_idx, H, pid
        else:
            # Grouped mode: all phenotypes in group must share ranges or we take union
            grouped = self.group_s.loc[phenotype_ids].groupby(self.group_s, sort=False)
            for k, (group_id, g) in enumerate(grouped, chr_offset + 1):
                if verbose:
                    _print_progress(k, self.n_groups, 'phenotype group')

                ids = list(g.index)
                idxs = [index_of[i] for i in ids]
                p = self._rows(self.phenotype_df, idxs)

                # Union of explicit indices (keeps genomic order from per-chrom pos sort)
                v_lists = [self.cis_v_idx[i] for i in ids if i in self.cis_v_idx]
                if not v_lists:
                    continue
                v_idx = np.unique(np.concatenate(v_lists)).astype(int, copy=False)

                G = self._rows(self.genotype_df, v_idx)
                G_idx = v_idx

                H_slice = self.haplotypes[v_idx, :, :]
                if self.on_the_fly_impute:
                    H_block = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)
                    H = self._interpolate_block(H_block)
                else:
                    H = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)
                if H is not None and H.ndim == 3 and H.shape[2] == 1:
                    H = H.reshape(H.shape[0], H.shape[1])

                yield p, G, G_idx, H, ids, group_id


# ----------------------------
# Helpers functions
# ----------------------------
def _to_pandas(df: Union[cuDF, pd.DataFrame, cudf.Series, pd.Series]) -> pd.DataFrame | pd.Series:
    return df.to_pandas() if isinstance(df, (cuDF, cudf.Series)) else df


def _get_sample_ids(df: Union[cuDF, pd.DataFrame]) -> List[str]:
    if isinstance(df, cuDF):
        return df["sample_id"].to_arrow().to_pylist()
    return df["sample_id"].tolist()


def _print_progress(k: int, n: int, entity: str) -> None:
    msg = f"\r    processing {entity} {k}/{n}"
    if k == n:
        msg += "\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
