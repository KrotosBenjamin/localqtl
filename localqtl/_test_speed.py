#!/usr/bin/env python3
import argparse
import time
import torch
import numpy as np
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity

# import functions under test
from _nominal import _process_phenotype_window
from core import calculate_corr_paired


# ------------------------
# Dummy helper objects
# ------------------------
class DummyIGC:
    def __init__(self, n_pheno, start=1_000_000):
        self.phenotype_start = {f"pheno{i}": start for i in range(n_pheno)}
        self.phenotype_end = {f"pheno{i}": start + 100 for i in range(n_pheno)}


def make_mapping_state(n_variants, device):
    return dict(
        device=device,
        af_all=np.random.rand(n_variants),
        maf_all=np.random.rand(n_variants),
        ma_samples_all=np.random.randint(5, 20, n_variants),
        ma_count_all=np.random.randint(5, 50, n_variants),
    )

# ------------------------
# CLI arguments
# ------------------------
parser = argparse.ArgumentParser(description="Profile _process_phenotype_window")
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--variants", type=int, default=50_000)
parser.add_argument("--samples", type=int, default=500)
parser.add_argument("--phenotypes", type=int, default=10)
parser.add_argument("--haps", type=int, default=2)
parser.add_argument("--window", type=int, default=500,
                    help="Number of variants per phenotype window")
args = parser.parse_args()

n_variants   = args.variants
n_samples    = args.samples
n_phenotypes = args.phenotypes
k_haps       = args.haps
window       = args.window
device       = args.device

torch.manual_seed(42)
np.random.seed(42)

# ------------------------
# Synthetic data
# ------------------------
# Genotypes: (variants x samples)
genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch.float32)

# Haplotypes: (variants x samples x k)
haplotypes = torch.rand(n_variants, n_samples, k_haps, dtype=torch.float32)

# Phenotypes: one vector per phenotype
phenotypes = [torch.randn(n_samples, dtype=torch.float32) for _ in range(n_phenotypes)]

# DataFrames
variant_ids = [f"var{i}" for i in range(n_variants)]
variant_df = pd.DataFrame({
    "pos": np.arange(n_variants) + 1,
}, index=variant_ids)

phenotype_ids = [f"pheno{i}" for i in range(n_phenotypes)]
phenotype_pos_df = pd.DataFrame({
    "chr": ["1"] * n_phenotypes,
    "start": np.arange(n_phenotypes) * 1000 + 1,
    "end": np.arange(n_phenotypes) * 1000 + 100,
}, index=phenotype_ids)

covariates_df = pd.DataFrame(np.random.randn(n_samples, 3),
                             columns=["cov1", "cov2", "cov3"])

# mapping_state
mapping_state = make_mapping_state(n_variants, device)

# ------------------------
# Build rows with fixed-size windows
# ------------------------
rows = []
for pid, pheno in zip(phenotype_ids, phenotypes):
    # pick a contiguous window start
    start = np.random.randint(0, n_variants - window)
    g_idx = np.arange(start, start + window)

    geno_slice = genotypes[g_idx]
    #hap_slice = haplotypes[g_idx]
    hap_slice  = torch.zeros((window, n_samples, 1), dtype=torch.float32)
    rows.append((pheno, geno_slice, g_idx, hap_slice, pid))

# genotype_ix_t: index tensor just covers the window
genotype_ix_t = None


# ------------------------
# Profile run
# ------------------------
def run_profile():
    print("\nProfiling _process_phenotype_window...")
    start = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("_process_phenotype_window"):
            out = _process_phenotype_window(
                rows, DummyIGC(n_phenotypes), genotype_ix_t,
                variant_df, phenotype_pos_df, covariates_df,
                residualizer=None, paired_covs_df=None, interaction_t=None,
                maf_threshold=0.01, interaction_df=None,
                maf_threshold_interaction=None, run_eigenmt=False,
                mapping_state=mapping_state
            )
        if device == "cuda":
            torch.cuda.synchronize()
        prof.step()
    end = time.time()

    print(f"Runtime: {end - start:.2f} sec")
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    return out


if __name__ == "__main__":
    result = run_profile()
    if result is None:
        print("No results (all filtered).")
    else:
        n_total, merged, _ = result
        print(f"\nProcessed {n_total} variant–phenotype pairs.")
        print("Merged keys:", list(merged.keys()))
