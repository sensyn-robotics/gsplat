#!/usr/bin/env python3
"""Tabulate gsplat experiment results.

Usage:
    uv run python scripts/compare_runs.py output/*/pipeline_results.json
    uv run python scripts/compare_runs.py output/exp_a output/exp_b ...

For each argument the script tries (in order):
  1. `<arg>` if it is a JSON file → reads it
  2. `<arg>/pipeline_results.json` → reads it
  3. `<arg>/gsplat/stats/val_step*.json` → reads the one with the highest step

Sorts the table by PSNR descending and writes a markdown copy to
`output/comparison_<UTC timestamp>.md`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _final_val_json(run_dir: Path) -> Path | None:
    stats = run_dir / "gsplat" / "stats"
    if not stats.is_dir():
        return None
    vals = list(stats.glob("val_step*.json"))
    if not vals:
        return None
    return max(vals, key=lambda p: int(p.stem.removeprefix("val_step")))


def _load_metrics(arg: str) -> dict | None:
    p = Path(arg)
    summary: dict | None = None
    if p.is_file() and p.suffix == ".json":
        summary = json.loads(p.read_text())
        run_dir = p.parent
    else:
        run_dir = p
        pr = run_dir / "pipeline_results.json"
        if pr.is_file():
            summary = json.loads(pr.read_text())

    final = (summary or {}).get("final_metrics") if summary else None
    if not final:
        vjson = _final_val_json(run_dir)
        if vjson is None:
            return None
        final = json.loads(vjson.read_text())

    refinement = (summary or {}).get("colmap_refinement") if summary else None
    n_reg = refinement.get("n_images") if isinstance(refinement, dict) else None
    return {
        "name": run_dir.name,
        "psnr": float(final.get("psnr", float("nan"))),
        "ssim": float(final.get("ssim", float("nan"))),
        "lpips": float(final.get("lpips", float("nan"))),
        "num_GS": int(final.get("num_GS", 0)),
        "reg_imgs": n_reg,
    }


def _render_table(rows: list[dict]) -> str:
    headers = ["name", "PSNR", "SSIM", "LPIPS", "num_GS", "reg_imgs"]
    widths = [max(len(h), max(len(_fmt(r, h)) for r in rows)) for h in headers]
    sep = "  ".join("-" * w for w in widths)
    head = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines = [head, sep]
    for r in rows:
        lines.append("  ".join(_fmt(r, h).ljust(w) for h, w in zip(headers, widths)))
    return "\n".join(lines)


def _fmt(row: dict, header: str) -> str:
    key = {"name": "name", "PSNR": "psnr", "SSIM": "ssim",
           "LPIPS": "lpips", "num_GS": "num_GS", "reg_imgs": "reg_imgs"}[header]
    v = row.get(key)
    if v is None:
        return "—"
    if header == "name":
        return str(v)
    if header in ("PSNR",):
        return f"{v:.3f}"
    if header in ("SSIM", "LPIPS"):
        return f"{v:.4f}"
    if header == "num_GS":
        return f"{v:,}"
    if header == "reg_imgs":
        return str(v)
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="run dirs or pipeline_results.json files")
    ap.add_argument("--out-md", type=Path, default=None, help="explicit output md path")
    args = ap.parse_args()

    rows = []
    skipped = []
    for arg in args.paths:
        m = _load_metrics(arg)
        if m is None:
            skipped.append(arg)
        else:
            rows.append(m)
    rows.sort(key=lambda r: r["psnr"], reverse=True)

    if not rows:
        print("No runs with metrics found.", file=sys.stderr)
        return 1

    table = _render_table(rows)
    print(table)
    if skipped:
        print(f"\n(skipped {len(skipped)} arg(s) with no metrics: {skipped})", file=sys.stderr)

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%SZ")
    out = args.out_md or Path("output") / f"comparison_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Run comparison (sorted by PSNR desc)\n\n```\n" + table + "\n```\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
