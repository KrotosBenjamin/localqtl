#!/usr/bin/env python3
import argparse
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# ------------------------
# CLI arguments
# ------------------------
parser = argparse.ArgumentParser(description="Profile calculate_corr_paired with flexible phenotypes")
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--variants", type=int, default=50_000)
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--phenotypes", type=int, default=64,
                    help="Number of shared phenotypes if --mode=batch")
parser.add_argument("--haps", type=int, default=2)
parser.add_argument("--mode", default="batch", choices=["batch", "per_variant", "both"],
                    help="Phenotype layout: batch=(p,s), per_variant=(v,s), both=run both")
parser.add_argument("--logdir", default="./profiler_log",
                    help="Directory for TensorBoard traces")
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
if k_haps == 2:  # collapse collinear haplotype
    H_t = H_t[:, :, :1]
    k_haps = 1


# ------------------------
# Regression function
# ------------------------
def calculate_corr_paired_zstack(G_t, H_t, Y_t):
    """
    Unified regression:
      - Y_t (p,s)  : shared phenotypes across variants
      - Y_t (v,s)  : per-variant phenotypes
      - Y_t (v,p,s): p phenotypes per variant
    """
    n_variants, n_samples = G_t.shape
    _, _, k = H_t.shape

    # Normalize Y_t -> (v,p,s)
    if Y_t.dim() == 2:
        if Y_t.shape[0] == n_variants:   # (v,s)
            Y_t = Y_t.unsqueeze(1)       # (v,1,s)
        else:                            # (p,s)
            Y_t = Y_t.unsqueeze(0).expand(n_variants, -1, -1)  # (v,p,s)
    elif Y_t.dim() == 3:
        pass
    else:
        raise ValueError("Y_t must be (p,s), (v,s), or (v,p,s)")

    n_pheno = Y_t.shape[1]

    # Precompute scalars
    sum_h = H_t.sum(1)                   # (v,k)
    HtH   = H_t.transpose(1, 2) @ H_t    # (v,k,k)
    sum_g  = G_t.sum(1)                  # (v,)
    sum_g2 = (G_t**2).sum(1)
    sum_gh = torch.bmm(G_t.unsqueeze(1), H_t).squeeze(1)

    XtX = torch.empty((n_variants, 2 + k, 2 + k),
                      device=G_t.device, dtype=G_t.dtype)
    XtX[:, 0, 0]   = n_samples
    XtX[:, 0, 1]   = XtX[:, 1, 0] = sum_g
    XtX[:, 1, 1]   = sum_g2
    XtX[:, 0, 2:]  = XtX[:, 2:, 0] = sum_h
    XtX[:, 1, 2:]  = XtX[:, 2:, 1] = sum_gh
    XtX[:, 2:, 2:] = HtH

    # XtY
    sum_y  = Y_t.sum(2)                                # (v,p)
    sum_gy = torch.einsum("vs,vps->vp", G_t, Y_t)      # (v,p)
    sum_hy = torch.einsum("vsk,vps->vpk", H_t, Y_t)    # (v,p,k)

    XtY = torch.zeros((n_variants, n_pheno, 2+k), device=G_t.device)
    XtY[:, :, 0] = sum_y
    XtY[:, :, 1] = sum_gy
    XtY[:, :, 2:] = sum_hy

    XtX = XtX.unsqueeze(1).expand(-1, n_pheno, -1, -1)  # (v,p,2+k,2+k)
    XtY = XtY.unsqueeze(-1)                             # (v,p,2+k,1)

    # Solve
    L = torch.linalg.cholesky(XtX)
    rhs = XtY.squeeze(-1).transpose(-1, -2)
    Z = torch.linalg.solve_triangular(L, rhs, upper=False)
    beta = torch.linalg.solve_triangular(L.transpose(-1,-2), Z, upper=True)
    return beta.transpose(-1,-2)  # (v,p,2+k)


# ------------------------
# Profiling helper
# ------------------------
def run_profile(Y_t, tag):
    print(f"\nProfiling mode={tag}...")
    start = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(f"{args.logdir}/{tag}")
    ) as prof:
        # Warm-up
        for _ in range(3):
            _ = calculate_corr_paired_zstack(G_t, H_t, Y_t)
        torch.cuda.synchronize()

        # Timed run
        with record_function("calculate_corr_paired_zstack"):
            beta = calculate_corr_paired_zstack(G_t, H_t, Y_t)
        torch.cuda.synchronize()
        prof.step()
    end = time.time()

    print(f"Runtime ({tag}): {end - start:.2f} sec")
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    return beta


# ------------------------
# Run
# ------------------------
if args.mode in ["batch", "both"]:
    Y_batch = torch.randn(n_phenotypes, n_samples, device=device)
    run_profile(Y_batch, "batch")

if args.mode in ["per_variant", "both"]:
    Y_var = torch.randn(n_variants, n_samples, device=device)
    run_profile(Y_var, "per_variant")

print(f"\nTrace written to {args.logdir}. Run `tensorboard --logdir={args.logdir}` to explore.")
