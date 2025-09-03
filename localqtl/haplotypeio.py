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

arr_mod = cp if cp.is_available() else np
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
        self.variant_ids = variant_df.index.to_numpy()

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

    def load_haplotypes(self) -> np.ndarray:
        """Force-load haplotype ancestry into memory as NumPy array."""
        return np.array(self.haplotypes)

    # @staticmethod
    # def _filter_zarr(zarr_in: str, zarr_out: str, indices: np.ndarray,
    #                  chunk_size: int = 10_000):
    #     """Write a filtered Zarr containing only rows at given indices."""
    #     daz = from_zarr(zarr_in)
    #     dst = daz[indices, :, :]
    #     dst = dst.rechunk((chunk_size, -1, -1))
    #     dst.to_zarr(zarr_out, overwrite=True)

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
        lb = bisect.bisect_left(chr_variant_dfs[chrom]['pos'].values, pos['start'] - window)
        ub = bisect.bisect_right(chr_variant_dfs[chrom]['pos'].values, pos['end'] + window)
        variant_r = chr_variant_dfs[chrom]['index'].values[[lb, ub -1]] if lb != ub else []

        has_variants = len(variant_r) > 0
        
        if has_variants:
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
    ):
        # Store
        self.genotype_df = genotype_df
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(self.variant_df.shape[0])

        self.loci_df = loci_df.copy()
        self.loci_df['index'] = np.arange(self.loci_df.shape[0])
        self.haplotypes = haplotypes  # Keep Zarr array

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
    def _interpolate_block(block: "arr_mod.ndarray") -> "arr_mod.ndarray":
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
        block_imputed = block.copy()
        loci_dim, sample_dim, ancestry_dim = block.shape

        for s in range(sample_dim):
            for a in range(ancestry_dim):
                col = block[:, s, a]
                mask = arr_mod.isnan(col)
                if arr_mod.any(mask):
                    idx = arr_mod.arange(loci_dim)
                    valid = ~mask
                    if arr_mod.any(valid):
                        # Linear interpolation and rounding
                        interpolated = arr_mod.round(
                            arr_mod.interp(idx[mask], idx[valid], col[valid])
                        )
                        col[mask] = interpolated.astype(int)
                block_imputed[:, s, a] = col
        return block_imputed

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
        self, chrom: Optional[str] = None,
        verbose: bool = False, as_cupy: bool = True,
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
                r = self.cis_ranges[pid]

                # Variant  and haplotype slice
                v_lb, v_ub = r if r is not None else (None, None)
                G = self._slice_rows(self.genotype_df, v_lb, (v_ub + 1) if v_ub is not None else None)
                G_idx = np.arange(v_lb, v_ub + 1) if v_lb is not None else np.arange(0, 0, dtype=int)

                H = None
                if v_lb is not None and v_ub is not None:
                    H_slice = self.haplotypes[v_lb:v_ub + 1, :, :] # dask array slice
                    H_block = H_slice.compute()
                    H = self._interpolate_block(H_block)

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

                # Validate identical ranges; if not, take union
                ranges = [self.cis_ranges[i] for i in ids]
                v_lbs = [r[0] for r in ranges if r is not None]
                v_ubs = [r[1] for r in ranges if r is not None]

                v_lb, v_ub = (min(v_lbs), max(v_ubs)) if len(v_lbs) else (None, None)

                G = self._slice_rows(self.genotype_df, v_lb, (v_ub + 1) if v_ub is not None else None) if v_lb is not None else None
                G_idx = np.arange(v_lb, v_ub + 1) if v_lb is not None else np.arange(0, 0, dtype=int)

                H = None
                if v_lb is not None and v_ub is not None:
                    H_slice = self.haplotypes[v_lb:v_ub + 1, :, :] # dask array slice
                    H_block = H_slice.compute()
                    H = self._interpolate_block(H_block)

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


# def _slice_rows(df_or_da, lb: Optional[int], ub: Optional[int], as_cupy: bool = True):
#     if lb is None or ub is None or lb < 0 or ub <= lb:
#         return None
#     # Dask array
#     if isinstance(df_or_da, da.Array):
#         out = df_or_da[lb:ub].compute()
#         return cp.asarray(out) if as_cupy else out
#     # cuDF
#     if isinstance(df_or_da, cuDF):
#         view = df_or_da.iloc[lb:ub]
#         return view.to_cupy() if as_cupy else view
#     # pandas DataFrame
#     view = df_or_da.iloc[lb:ub].to_numpy(copy=False)
#     return cp.asarray(view) if as_cupy else view

# def _row(df_or_da, i: int, as_cupy: bool = True):
#     if isinstance(df_or_da, da.Array):
#         out = df_or_da[i].compute()
#         return cp.asarray(out) if as_cupy else out
#     if isinstance(df_or_da, cuDF):
#         arr = df_or_da.iloc[i]
#         return arr.to_cupy() if as_cupy else arr
#     arr = df_or_da.iloc[i].to_numpy(copy=False)
#     return cp.asarray(arr) if as_cupy else arr

#     @staticmethod
#     def _rows(df_or_da, idxs: List[int], as_cupy: bool = True):
#         if isinstance(df_or_da, da.Array):
#             out = df_or_da[idxs].compute()
#             return cp.asarray(out) if as_cupy else out
#         if isinstance(df_or_da, cuDF):
#             arr = df_or_da.iloc[idxs]
#             return arr.to_cupy() if as_cupy else arr
#         arr = df_or_da.iloc[idxs].to_numpy(copy=False)
#         return cp.asarray(arr) if as_cupy else arr

        # # Ensure unique variant positions
        # variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"],
        #                                         keep="first").copy()

        # # Align variant grid
        # variant_loci = (
        #     variant_df.merge(_to_pandas(loci), on=["chrom", "pos"], how="outer",
        #                      indicator=True)
        #     .loc[:, ["chrom", "pos", "i", "_merge"]]
        # )
        # present_mask = ~(variant_loci["_merge"] == "right_only")
        # keep_idx = np.where(present_mask.values)[0]

        # # Impute and load zarr
        # zarr_file = f"{self.zarr_dir}/local-ancestry.zarr"
        # zarr_masked = f"{self.zarr_dir}/local-ancestry.masked.zarr"
        # if (not exists(zarr_file)) or impute:
        #     _ = interpolate_array(variant_loci, admix, self.zarr_dir)

        # if (not exists(zarr_masked)) or impute:
        #     self._filter_zarr(zarr_file, zarr_masked, keep_idx)
        # self.admix = zarr.open_array(zarr_masked, mode='r')  # (variants_aligned x samples x pops)

        # # Build filtered loci table
        # filtered = variant_loci.loc[present_mask].copy().drop(["i", "_merge"],
        #                                                       axis=1).reset_index(drop=True)
        # self.loci = cudf.from_pandas(filtered)
