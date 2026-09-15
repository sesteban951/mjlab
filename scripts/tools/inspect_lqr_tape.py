"""Plot an LQR tape's feedback gains and cost-to-go conditioning over time.

uv run python scripts/tools/inspect_lqr_tape.py trajectories/single/lqr_sideroll/lqr_sideroll_motion_prior.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("tape", type=Path)
  ap.add_argument("--out", type=Path, default=None)
  ap.add_argument("--top", type=int, default=8, help="actuators to label by peak gain")
  args = ap.parse_args()

  d = np.load(args.tape, allow_pickle=True)
  K = d["gain"].astype(np.float64)  # (T, nu, ndx)
  ff = d["feedforward"].astype(np.float64)
  names = [str(s) for s in d["joint_names"]]
  fps = float(np.atleast_1d(d["fps"])[0])
  t = np.arange(K.shape[0]) / fps
  ndx = K.shape[2]
  nq = ndx // 2  # tangent layout is [dq(nq), dv(nq)]

  row = np.linalg.norm(K, axis=2)  # per-actuator gain magnitude
  col = np.linalg.norm(K, axis=1)  # per-feature gain magnitude
  sv = np.linalg.svd(K, compute_uv=False)

  has_p = "P" in d.files
  if has_p:
    P = d["P"].astype(np.float64)
    P = 0.5 * (P + P.transpose(0, 2, 1))
    w = np.linalg.eigvalsh(P)[:, ::-1]
    dg = np.einsum("tii->ti", P)
    Dm = 1.0 / np.sqrt(np.clip(dg, 1e-30, None))
    cond_jacobi = np.linalg.cond(P * Dm[:, :, None] * Dm[:, None, :])

  nrows = 3 if has_p else 2
  fig, ax = plt.subplots(nrows, 2, figsize=(15, 4 * nrows), constrained_layout=True)

  a = ax[0, 0]
  for i in np.argsort(-row.max(axis=0))[: args.top]:
    a.plot(t, row[:, i], lw=1, label=names[i])
  a.plot(t, row.mean(axis=1), "k--", lw=1, label="mean over actuators")
  a.set(title="per-actuator gain row norm", xlabel="s", ylabel="||K[i,:]||")
  a.legend(fontsize=6, ncol=2)

  a = ax[0, 1]
  im = a.imshow(
    np.log10(row.T + 1e-6),
    aspect="auto",
    origin="lower",
    extent=(t[0], t[-1], -0.5, len(names) - 0.5),
    cmap="magma",
  )
  a.set(title="log10 ||K[i,:]|| (all actuators)", xlabel="s")
  a.set_yticks(range(len(names)))
  a.set_yticklabels([n.replace("_joint", "") for n in names], fontsize=5)
  fig.colorbar(im, ax=a)

  a = ax[1, 0]
  a.semilogy(t, sv[:, 0], label="sigma_max(K)")
  a.semilogy(t, sv[:, -1], label="sigma_min(K)")
  a.semilogy(t, np.linalg.norm(K, axis=(1, 2)), label="||K||_F")
  a.semilogy(t, np.abs(K).max(axis=(1, 2)), label="max |K_ij|")
  a.set(title="gain magnitude / rank health", xlabel="s")
  a.legend(fontsize=7)

  a = ax[1, 1]
  a.plot(
    t, np.linalg.norm(col[:, :nq], axis=1), label="position features ||K[:, :nq]||"
  )
  a.plot(
    t, np.linalg.norm(col[:, nq:], axis=1), label="velocity features ||K[:, nq:]||"
  )
  a.plot(t, np.linalg.norm(ff, axis=1), label="||feedforward||")
  a.set(title="feature-block split and feedforward", xlabel="s")
  a.legend(fontsize=7)

  if has_p:
    a = ax[2, 0]
    a.semilogy(t, np.linalg.cond(P), label="cond(P)")
    a.semilogy(t, cond_jacobi, label="cond(P) after diagonal scaling")
    a.axhline(1 / np.finfo(np.float32).eps, color="r", ls=":", lw=1, label="1/eps_f32")
    a.set(title="cost-to-go conditioning", xlabel="s")
    a.legend(fontsize=7)

    a = ax[2, 1]
    for k, lbl in [
      (0, "lambda_max"),
      (4, "lambda_5"),
      (34, "lambda_35"),
      (-1, "lambda_min"),
    ]:
      a.semilogy(t, w[:, k], lw=1, label=lbl)
    a.set(title="eigenvalues of P", xlabel="s")
    a.legend(fontsize=7)

  out = args.out or args.tape.with_suffix(".gains.png")
  fig.savefig(out, dpi=140)
  print(f"wrote {out}")

  print(f"\nT={K.shape[0]}  nu={K.shape[1]}  ndx={ndx}  fps={fps}")
  print(
    f"||K||_F: med {np.median(np.linalg.norm(K, axis=(1, 2))):.4g}  max {np.linalg.norm(K, axis=(1, 2)).max():.4g}"
  )
  print(f"max |K_ij| = {np.abs(K).max():.4g}")
  rank = (sv > sv[:, :1] * 1e-10).sum(axis=1)
  drop = np.where(rank < K.shape[1])[0]
  print(f"rank-deficient frames: {len(drop)} {drop[:10]}")
  for f in drop[:10]:
    dead = [names[i] for i in np.where(row[f] < 1e-12)[0]]
    print(f"  t={f} ({f / fps:.2f}s) zero-gain actuators: {dead}")
  rel = (
    np.linalg.norm(np.diff(K, axis=0), axis=(1, 2))
    / np.linalg.norm(K, axis=(1, 2))[:-1]
  )
  print(
    f"frame-to-frame ||dK||/||K||: med {np.median(rel):.3f}  p95 {np.percentile(rel, 95):.3f}  max {rel.max():.3f} at t={rel.argmax()}"
  )
  if has_p:
    c = np.linalg.cond(P)
    print(
      f"cond(P): med {np.median(c):.3e}  max {c.max():.3e}   after diag scaling: med {np.median(cond_jacobi):.3e}"
    )
    print(
      f"eig(P) in [{w.min():.4g}, {w.max():.4g}]; min/max = {w.min() / w.max():.3g} vs f32 eps {np.finfo(np.float32).eps:.3g}"
    )
    print(f"negative-eigenvalue frames: {(w.min(axis=1) < 0).sum()}")
    tr = w.sum(1)
    print(
      "trace share: "
      + "  ".join(
        f"top-{k}={100 * np.median(w[:, :k].sum(1) / tr):.1f}%" for k in (1, 3, 5, 10)
      )
    )


if __name__ == "__main__":
  main()
