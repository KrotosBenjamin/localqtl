#!/usr/bin/env python3
import argparse
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from core import calculate_corr_paired


# ------------------------
# CLI arguments
# ------------------------
parser = argparse.ArgumentParser(description="Profile calculate_corr_paired directly")
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--variants", type=int, default=50_000,
                    help="Number of variants")
parser.add_argument("--samples", type=int, default=500,
                    help="Number of samples")
parser.add_argument("--phenotypes", type=int, default=10,
                    help="Number of phenotypes")
parser.add_argument("--haps", type=int, default=1,
                    help="Number of haplotype covariates (k)")
args = parser.parse_args()

n_variants   = args.variants
n_samples    = args.samples
n_pheno      = args.phenotypes
k_haps       = args.haps
device       = args.device

torch.manual_seed(42)


# ------------------------
# Synthetic data
# ------------------------
# Genotypes (variants × samples)
G_t = torch.randint(0, 3, (n_variants, n_samples),
                    device=device, dtype=torch.float32)

# Haplotypes (variants × samples × k)
H_t = torch.rand(n_variants, n_samples, k_haps,
                 device=device, dtype=torch.float32)

# Phenotypes (phenotypes × samples)
Y_t = torch.randn(n_pheno, n_samples, device=device)


# ------------------------
# Profile run
# ------------------------
def run_profile():
    print("\nProfiling calculate_corr_paired...")
    start = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("calculate_corr_paired"):
            out = calculate_corr_paired(
                G_t, H_t, Y_t,
                residualizer=None,
                use_pinv=False,
                dof_vector=None,
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
    beta_g, beta_h, tstat_g, se_g, se_h = result
    print("\nOutput shapes:")
    print("  beta_g:", beta_g.shape)
    print("  beta_h:", beta_h.shape)
    print("  tstat_g:", tstat_g.shape)
    print("  se_g:", se_g.shape)
    print("  se_h:", se_h.shape)
