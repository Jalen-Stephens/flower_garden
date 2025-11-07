# Analyzer for Group 5 benchmarking results
# Moved from root benchmarks/ for focused group5 development

import csv
import os
from statistics import mean, median
from collections import defaultdict, OrderedDict

def pick_col(row, header, candidates):
    for name in candidates:
        if name in header:
            return name
    for key in header:
        lk = key.lower()
        for frag in candidates:
            if frag.lower() in lk:
                return key
    return None

def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def analyze(in_path: str, out_csv: str, out_md: str):
    if not os.path.exists(in_path):
        print(f"No results file found at {in_path}")
        return 1
    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            print("No rows to analyze")
            return 0
        header = reader.fieldnames or []
    col_variant = pick_col(rows[0], header, ["variant"]) or "variant"
    col_config = pick_col(rows[0], header, ["config", "config_path", "json_path"]) or "config"
    col_g5 = pick_col(rows[0], header, ["g5_growth", "growth_g5", "growth"]) or "growth"
    col_g2 = pick_col(rows[0], header, ["g2_growth", "growth_g2"])  # may be None
    col_g3 = pick_col(rows[0], header, ["g3_growth", "growth_g3"])  # may be None
    col_plants = pick_col(rows[0], header, ["plants", "plants_placed"]) or "plants"
    col_time = pick_col(rows[0], header, ["time", "placement_time", "runtime_sec"]) or "time"
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r.get(col_variant, "default")].append(r)
    summary = []
    for variant, vr in by_variant.items():
        g5_vals = [to_float(r.get(col_g5, 0)) for r in vr]
        plants_vals = [to_float(r.get(col_plants, 0)) for r in vr]
        time_vals = [to_float(r.get(col_time, 0)) for r in vr]
        if col_g2:
            g2_vals = [to_float(r.get(col_g2, 0)) for r in vr]
            d_g2 = [g5 - g2 for g5, g2 in zip(g5_vals, g2_vals)]
        else:
            d_g2 = []
        if col_g3:
            g3_vals = [to_float(r.get(col_g3, 0)) for r in vr]
            d_g3 = [g5 - g3 for g5, g3 in zip(g5_vals, g3_vals)]
        else:
            d_g3 = []
        entry = OrderedDict(
            variant=variant,
            runs=len(vr),
            avg_growth_g5=round(mean(g5_vals), 2),
            med_growth_g5=round(median(g5_vals), 2),
            avg_plants=round(mean(plants_vals), 2),
            med_plants=round(median(plants_vals), 2),
            avg_time_s=round(mean(time_vals), 4),
            med_time_s=round(median(time_vals), 4),
            avg_delta_vs_g2=round(mean(d_g2), 2) if d_g2 else None,
            med_delta_vs_g2=round(median(d_g2), 2) if d_g2 else None,
            avg_delta_vs_g3=round(mean(d_g3), 2) if d_g3 else None,
            med_delta_vs_g3=round(median(d_g3), 2) if d_g3 else None,
        )
        summary.append(entry)
    def sort_key(e):
        d2 = e.get("avg_delta_vs_g2")
        return (d2 if d2 is not None else float("-inf"), e["avg_growth_g5"]) 
    summary.sort(key=sort_key, reverse=True)
    fieldnames = list(summary[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Benchmark Summary by Variant\n\n")
        f.write(f"Source: `{os.path.relpath(in_path)}`\n\n")
        f.write("| variant | runs | avg_growth_g5 | avg_delta_vs_g2 | avg_delta_vs_g3 | avg_plants | avg_time_s |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['variant']} | {row['runs']} | {row['avg_growth_g5']} | {row.get('avg_delta_vs_g2','')} | {row.get('avg_delta_vs_g3','')} | {row['avg_plants']} | {row['avg_time_s']} |\n"
            )
    print("Top 5 variants by avg_delta_vs_g2 then avg_growth_g5:")
    for r in summary[:5]:
        print(
            f" - {r['variant']}: runs={r['runs']}, avg_g5={r['avg_growth_g5']}, dG2={r.get('avg_delta_vs_g2')}, dG3={r.get('avg_delta_vs_g3')}, time={r['avg_time_s']}s"
        )
    print(f"\nWrote {out_csv} and {out_md}")
    return 0

if __name__ == "__main__":
    base = os.path.dirname(__file__)
    in_path = os.path.join(base, "results_compare.csv")
    out_csv = os.path.join(base, "summary.csv")
    out_md = os.path.join(base, "summary.md")
    raise SystemExit(analyze(in_path, out_csv, out_md))
