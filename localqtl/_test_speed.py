#!/usr/bin/env python3
import argparse
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

# ------------------------
# CLI arguments
# ------------------------
parser = argparse.ArgumentParser(description="Profile calculate_corr_paired with batched phenotypes")
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--variants", type=int, default=100_000)
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--phenotypes", type=int, default=64)  # batch size of phenotypes
parser.add_argument("--haps", type=int, default=2)
args = parser.parse_args()

n_variants   = args.variants
n_samples    = args.samples
n_phenotypes = args.phenotypes
k_haps       = args.haps
device       = args.device

torch.manual_seed(42)

# ------------------------
# Generate synthetic data
# ------------------------
# Genotypes (variants × samples)
G_t = torch.randint(0, 3, (n_variants, n_samples),
                    device=device, dtype=torch.float32)

# Haplotypes (variants × samples × k)
H_t = torch.rand(n_variants, n_samples, k_haps, device=device)

# If k=2 (two ancestries), collapse to 1 covariate to avoid collinearity
if k_haps == 2:
    H_t = H_t[:, :, :1]
    k_haps = 1

# Phenotypes (batch of phenotypes × samples)
Y_t = torch.randn(n_phenotypes, n_samples, device=device)

print(f"Generated test data: {n_variants} variants × {n_samples} samples × {n_phenotypes} phenotypes")
print(f"Haplotype covariates: {k_haps}")

# ------------------------
# Dummy residualizer (identity)
# ------------------------
class DummyResidualizer:
    def transform(self, X):
        return X
residualizer = DummyResidualizer()

# ------------------------
# Batched regression function
# ------------------------
def calculate_corr_paired_batched(G_t, H_t, Y_t):
    """
    Batched regression: multiple phenotypes per variant.
    G_t: (n_variants, n_samples)
    H_t: (n_variants, n_samples, k)
    Y_t: (n_phenotypes, n_samples)
    """
    n_variants, n_samples = G_t.shape
    n_pheno = Y_t.shape[0]
    _, _, k = H_t.shape

    # Precompute haplotype-only constants
    sum_h = H_t.sum(1) # (n_variants, k)
    HtH   = H_t.transpose(1, 2) @ H_t

    # Genotype scalars
    sum_g  = G_t.sum(1) # (n_variants,)
    sum_g2 = (G_t**2).sum(1)
    sum_gh = torch.bmm(G_t.unsqueeze(1), H_t).squeeze(1)

    # Assemble XtX per variant
    XtX = torch.empty((n_variants, 2 + k, 2 + k),
                      device=G_t.device, dtype=G_t.dtype)
    XtX[:, 0, 0]   = n_samples
    XtX[:, 0, 1]   = XtX[:, 1, 0] = sum_g
    XtX[:, 1, 1]   = sum_g2
    XtX[:, 0, 2:]  = XtX[:, 2:, 0] = sum_h
    XtX[:, 1, 2:]  = XtX[:, 2:, 1] = sum_gh
    XtX[:, 2:, 2:] = HtH

    # Now build XtY for all phenotypes at once
    # G_t: (v,s), Y_t: (p,s), H_t: (v,s,k)
    sum_y  = Y_t.sum(1)                                 # (p,)
    sum_gy = G_t @ Y_t.T                                # (v,p)
    sum_hy = torch.einsum("vsk,ps->vpk", H_t, Y_t)      # (v,p,k)

    # XtY shape: (v, p, 2+k)
    XtY = torch.zeros((n_variants, n_pheno, 2+k), device=G_t.device)
    XtY[:, :, 0] = sum_y.unsqueeze(0).expand(n_variants, -1)
    XtY[:, :, 1] = sum_gy
    XtY[:, :, 2:] = sum_hy
    XtY = XtY.unsqueeze(-1)  # (v, p, 2+k, 1)

    # Solve per phenotype: vectorized
    # broadcast XtX: (v,1,2+k,2+k) vs XtY: (v,p,2+k,1)
    L = torch.linalg.cholesky(XtX)
    beta = torch.cholesky_solve(XtY, L.unsqueeze(1)).squeeze(-1)

    return beta

# ------------------------
# Profiling
# ------------------------
print("Starting profiling run...")
start = time.time()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("calculate_corr_paired_batched_test"):
        beta = calculate_corr_paired_batched(G_t, H_t, Y_t)

if device == "cuda":
    torch.cuda.synchronize()
end = time.time()

print(f"Runtime: {end - start:.2f} seconds")
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=15))
