#!/usr/bin/env python3
"""Train the Hyperstrike behavioral anti-cheat model and export it for the Rust server.

The model is a tiny MLP (~6k params) that runs on the server's single shared CPU in microseconds.
Training data:
  * synthetic subject-matches from `cargo run --release --example ac_datagen` (labels known)
  * optionally, real match reports exported from the admin panel (Anti-Cheat -> Export training
    data). Real players default to label 0 ("human") with a low weight, admin-cleared false
    positives get label 0 with a high weight, and confirmed ban-wave cheaters get label 1.

Usage:
  python tools/anticheat/train.py                                   # synthetic only
  python tools/anticheat/train.py --real path/to/ac_export.ndjson   # + real data
  python tools/anticheat/train.py --gen 6000                        # regenerate synthetic first

Writes rust/sim/assets/ac_model.json (embedded into the server binary at build time).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "rust" / "sim"
DATA = Path(__file__).resolve().parent / "data"
OUT = SIM / "assets" / "ac_model.json"


def feature_names() -> list[str]:
    src = (SIM / "src" / "anticheat.rs").read_text(encoding="utf-8")
    block = src.split("pub const FEATURE_NAMES: &[&str] = &[", 1)[1].split("];", 1)[0]
    return [line.strip().strip(",").strip('"') for line in block.splitlines() if line.strip().startswith('"')]


def run_datagen(path: Path, matches: int, seed: int) -> None:
    print(f"[gen] building + running ac_datagen ({matches} matches)...", flush=True)
    subprocess.run(
        ["cargo", "run", "--release", "--example", "ac_datagen", "--", str(path), str(matches), str(os.cpu_count() or 8), str(seed)],
        cwd=SIM,
        check=True,
    )


def load_synthetic(path: Path, n_feat: int):
    X, y, w, kinds, ev = [], [], [], [], []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if len(r["features"]) != n_feat or r["evidence"].get("live_secs", 0) < 60:
            continue
        X.append(r["features"])
        y.append(r["label"])
        w.append(1.0)
        kinds.append(r["kind"])
        ev.append(r["evidence"])
    return X, y, w, kinds, ev


def load_real(path: Path, n_feat: int):
    X, y, w, kinds, ev = [], [], [], [], []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        f = r.get("features") or []
        if len(f) != n_feat or (r.get("liveSecs") or 0) < 60:
            continue
        lab = r.get("label")
        if lab == 1:
            y.append(1); w.append(2.0); kinds.append("real_banned")
        elif lab == 0:
            y.append(0); w.append(3.0); kinds.append("real_cleared")
        else:
            y.append(0); w.append(0.5); kinds.append("real_unlabeled")
        X.append(f)
        ev.append(r.get("evidence") or {})
    return X, y, w, kinds, ev


class Net(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.l1 = nn.Linear(d, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(torch.relu(self.l1(x)))
        x = self.drop(torch.relu(self.l2(x)))
        return self.l3(x).squeeze(-1)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=str(DATA / "synthetic.jsonl"))
    ap.add_argument("--real", action="append", default=[], help="admin-panel export (.ndjson); repeatable")
    ap.add_argument("--gen", type=int, default=0, help="regenerate synthetic data with N matches first")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    names = feature_names()
    n_feat = len(names)
    syn = Path(args.synthetic)
    if args.gen:
        DATA.mkdir(parents=True, exist_ok=True)
        run_datagen(syn, args.gen, args.seed)

    X, y, w, kinds, ev = load_synthetic(syn, n_feat)
    print(f"[data] synthetic: {len(X)} rows  {dict(Counter(kinds))}")
    for rp in args.real:
        rx, ry, rw, rk, rev = load_real(Path(rp), n_feat)
        print(f"[data] real {rp}: {len(rx)} rows  {dict(Counter(rk))}")
        X += rx; y += ry; w += rw; kinds += rk; ev += rev

    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    kinds = np.asarray(kinds)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    idx = rng.permutation(len(X))
    n_val, n_test = int(0.12 * len(X)), int(0.12 * len(X))
    te, va, tr = idx[:n_test], idx[n_test : n_test + n_val], idx[n_test + n_val :]

    mean = X[tr].mean(0)
    std = X[tr].std(0)
    std = np.where(std < 1e-4, 1.0, std).astype(np.float32)
    norm = lambda a: np.clip((a - mean) / std, -8, 8).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={dev} features={n_feat} train={len(tr)} val={len(va)} test={len(te)}")
    Xt = {k: torch.tensor(norm(X[v]), device=dev) for k, v in (("tr", tr), ("va", va), ("te", te))}
    yt = {k: torch.tensor(y[v], device=dev) for k, v in (("tr", tr), ("va", va), ("te", te))}
    wt = torch.tensor(w[tr], device=dev)
    # balance classes (after sample weights)
    pos_w = float((w[tr] * (1 - y[tr])).sum() / max((w[tr] * y[tr]).sum(), 1e-6))

    net = Net(n_feat).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_w, device=dev))
    best, best_state, bad = 1e9, None, 0
    bs = 512
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), bs):
            b = perm[i : i + bs]
            xb = Xt["tr"][b] + 0.08 * torch.randn_like(Xt["tr"][b])  # robustness to sim artifacts
            loss = (bce(net(xb), yt["tr"][b]) * wt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = bce(net(Xt["va"]), yt["va"]).mean().item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 50 == 0:
            print(f"  epoch {ep:4d}  val_loss {vl:.4f}")
        if bad > 80:
            print(f"  early stop at epoch {ep}")
            break
    net.load_state_dict(best_state)
    net.eval()

    # Temperature scaling on validation (folded into the last layer)
    with torch.no_grad():
        logits_va = net(Xt["va"])
    T = torch.ones(1, device=dev, requires_grad=True)
    topt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    plain = nn.BCEWithLogitsLoss()
    def closure():
        topt.zero_grad()
        l = plain(logits_va / T.clamp(min=0.05), yt["va"])
        l.backward()
        return l
    topt.step(closure)
    temp = float(T.clamp(min=0.05).item())

    with torch.no_grad():
        p_te = torch.sigmoid(net(Xt["te"]) / temp).cpu().numpy()
    y_te = y[te]
    k_te = kinds[te]
    metrics = {"auc": auc(p_te, y_te), "temperature": temp, "n_train": float(len(tr))}
    print(f"\n[test] AUC {metrics['auc']:.4f}  (temperature {temp:.3f})")
    humans = y_te == 0
    for th in (0.5, 0.8, 0.9, 0.95):
        fpr = float((p_te[humans] >= th).mean()) if humans.any() else float("nan")
        metrics[f"human_fpr_at_{th}"] = fpr
        line = f"  thr {th:.2f}: human FPR {fpr*100:6.2f}%"
        for k in sorted(set(k_te)):
            m = k_te == k
            if y_te[m].mean() == 1:
                tpr = float((p_te[m] >= th).mean())
                metrics[f"tpr_{k}_at_{th}"] = tpr
                line += f" | {k} {tpr*100:5.1f}%"
        print(line)

    # human reference ranges for the LLM dossier
    ref = {}
    hum_ev = [e for e, k in zip(ev, kinds) if k in ("human", "real_cleared", "real_unlabeled")]
    keys = sorted({k for e in hum_ev for k in e})
    for k in keys:
        vals = np.asarray([e[k] for e in hum_ev if k in e and isinstance(e[k], (int, float)) and math.isfinite(e[k])])
        if len(vals) >= 20:
            ref[k] = [float(np.percentile(vals, 5)), float(np.percentile(vals, 50)), float(np.percentile(vals, 95))]

    sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    layers = [
        {"w": sd["l1.weight"].tolist(), "b": sd["l1.bias"].tolist()},
        {"w": sd["l2.weight"].tolist(), "b": sd["l2.bias"].tolist()},
        {"w": (sd["l3.weight"] / temp).tolist(), "b": (sd["l3.bias"] / temp).tolist()},
    ]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    tag = "syn+real" if args.real else "syn"
    model = {
        "version": f"{tag}-{stamp}",
        "features": names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "layers": layers,
        "human_ref": ref,
        "metrics": {k: (float(v) if math.isfinite(v) else 0.0) for k, v in metrics.items()},
    }
    out = Path(args.out)
    out.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"\n[export] {out}  version={model['version']}  ({out.stat().st_size/1024:.0f} KB)")
    print("Rebuild + deploy the server to ship it:  fly deploy")


if __name__ == "__main__":
    main()
