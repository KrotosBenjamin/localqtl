#!/usr/bin/env python3
import argparse
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

# ------------------------
# CLI arguments
# ------------------------
parser = argparse.ArgumentParser(description="Profile calculate_corr_paired")
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--mode", default="single", choices=["single", "multi"],
                    help="single = (1, n_samples) phenotype, multi = (n_variants, n_samples)")
parser.add_argument("--variants", type=int, default=100_000)
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--phenotypes", type=int, default=200)
parser.add_argument("--haps", type=int, default=2)
args = parser.parse_args()

n_variants   = args.variants
n_samples    = args.samples
n_phenotypes = args.phenotypes
k_haps       = args.haps
device       = args.device

# ------------------------
# Generate synthetic data
# ------------------------
torch.manual_seed(42)

# Genotypes (variants × samples)
G_t = torch.randint(0, 3, (n_variants, n_samples), device=device, dtype=torch.float32)

# Haplotypes (variants × samples × k)
H_t = torch.rand(n_variants, n_samples, k_haps, device=device)

# If k=2 (two ancestries), collapse to 1 covariate to avoid collinearity
if k_haps == 2:
    H_t = H_t[:, :, :1]  # keep one ancestry column
    k_haps = 1

# Phenotypes
if args.mode == "single":
    # One phenotype for all variants
    Y_t = torch.randn(1, n_samples, device=device)
elif args.mode == "multi":
    # Each variant gets its own phenotype (stress test)
    Y_t = torch.randn(n_variants, n_samples, device=device)

print(f"Generated test data: {n_variants} variants × {n_samples} samples × mode={args.mode}")
print(f"Haplotype covariates: {k_haps}")

# ------------------------
# Dummy residualizer (identity)
# ------------------------
class DummyResidualizer:
    def transform(self, X):
        return X
residualizer = DummyResidualizer()

# ------------------------
# Function under test
# ------------------------
def calculate_corr_paired(
        G_t, H_t, Y_t, residualizer=None, use_pinv=False,
        return_se_h=True, dof_vector=None,
):
    n_variants, n_samples = G_t.shape
    if H_t.ndim == 2:
        H_t = H_t.unsqueeze(-1)
    _, _, k = H_t.shape

    # Precompute cross-products
    sum_h  = H_t.sum(1)                      # (v,k)
    HtH    = H_t.transpose(1, 2) @ H_t       # (v,k,k)
    sum_g  = G_t.sum(1)                      # (v,)
    sum_g2 = (G_t**2).sum(1)                 # (v,)
    sum_gh = torch.bmm(G_t.unsqueeze(1), H_t).squeeze(1)  # (v,k)

    # Assemble XtX
    XtX = torch.zeros((n_variants, 2+k, 2+k), device=G_t.device)
    XtX[:, 0, 0] = n_samples
    XtX[:, 0, 1] = XtX[:, 1, 0] = sum_g
    XtX[:, 1, 1] = sum_g2
    XtX[:, 0, 2:] = XtX[:, 2:, 0] = sum_h
    XtX[:, 1, 2:] = XtX[:, 2:, 1] = sum_gh
    XtX[:, 2:, 2:] = HtH

    # Assemble XtY
    if Y_t.shape[0] == 1:  # shared phenotype
        y = Y_t.view(-1)
        XtY = torch.cat([
            y.sum().expand(n_variants, 1),
            (G_t * y).sum(1, keepdim=True),
            torch.bmm(y.expand(n_variants, -1).unsqueeze(1), H_t).squeeze(1)
        ], dim=1).unsqueeze(-1)
    elif Y_t.shape[0] == n_variants:  # variant-specific phenotype
        XtY = torch.cat([
            Y_t.sum(1, keepdim=True),
            (G_t * Y_t).sum(1, keepdim=True),
            torch.bmm(Y_t.unsqueeze(1), H_t).squeeze(1)
        ], dim=1).unsqueeze(-1)
    else:
        raise ValueError(f"bad Y_t shape {Y_t.shape}")

    # Solve
    beta = torch.linalg.solve(XtX, XtY).squeeze(-1)
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
    with record_function("calculate_corr_paired_test"):
        beta = calculate_corr_paired(G_t, H_t, Y_t, residualizer=None)

if device == "cuda":
    torch.cuda.synchronize()
end = time.time()

print(f"Runtime: {end - start:.2f} seconds")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
