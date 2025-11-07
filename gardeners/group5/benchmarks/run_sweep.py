# Benchmark harness for Group 5 strategies
# Moved from root benchmarks/ for focused group5 development

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.runner import GameRunner  # noqa: E402
from gardeners.group5.gardener import Gardener5  # noqa: E402
from gardeners.group2.gardener import Gardener2  # noqa: E402
from gardeners.group3.gardener import Gardener3  # noqa: E402
from gardeners.group5.principled_strategy import (
    PrincipledStrategy,
)

@dataclass
class Variant:
    name: str
    overrides: Dict[str, Any]

def find_config_files() -> List[Path]:
    files: List[Path] = []
    gardeners_dir = ROOT / 'gardeners'
    for group in gardeners_dir.iterdir():
        if not group.is_dir():
            continue
        for cfg_dir_name in ('config', 'configs'):
            cfg_dir = group / cfg_dir_name
            if cfg_dir.is_dir():
                files.extend(sorted(cfg_dir.glob('*.json')))
    examples_dir = ROOT / 'examples'
    if examples_dir.is_dir():
        files.extend(sorted(examples_dir.glob('*.json')))
    uniq: Dict[str, Path] = {str(p.resolve()): p for p in files}
    return list(uniq.values())

def variants_set() -> List[Variant]:
    return [
        Variant('default', {}),
        Variant('gstep_low', {'GREEDY_GRID_STEP': 0.14}),
        Variant('gstep_high', {'GREEDY_GRID_STEP': 0.22}),
        Variant('jitter_low', {'LATTICE_JITTER': 0.10}),
        Variant('jitter_high', {'LATTICE_JITTER': 0.20}),
        Variant('triads_fewer', {'MAX_TRIADS': 40}),
        Variant('triads_more', {'MAX_TRIADS': 80}),
        Variant('inter_lo', {'INTERACTION_RADIUS_FACTOR': 0.95}),
        Variant('inter_hi', {'INTERACTION_RADIUS_FACTOR': 1.05}),
        Variant('cluster_cap', {'CLUSTER_LIMIT': 100}),
    ]

def apply_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    prev: Dict[str, Any] = {}
    for k, v in overrides.items():
        prev[k] = getattr(PrincipledStrategy, k)
        setattr(PrincipledStrategy, k, v)
    return prev

def restore_overrides(prev: Dict[str, Any]) -> None:
    for k, v in prev.items():
        setattr(PrincipledStrategy, k, v)

def run_one(config_path: Path, turns: int, gardener_cls) -> Dict[str, Any]:
    runner = GameRunner(varieties_file=str(config_path), simulation_turns=turns)
    result = runner.run(gardener_cls)
    return result

def plan_runs(configs: List[Path], variants: List[Variant], target: int) -> List[Tuple[Path, Variant]]:
    if not configs:
        return []
    planned: List[Tuple[Path, Variant]] = []
    idx_c, idx_v = 0, 0
    while len(planned) < target:
        planned.append((configs[idx_c % len(configs)], variants[idx_v % len(variants)]))
        idx_c += 1
        if idx_c % len(configs) == 0:
            idx_v += 1
    return planned

def main() -> None:
    ap = argparse.ArgumentParser(description='Benchmark Group 5 strategy across configs and variants')
    ap.add_argument('--turns', type=int, default=1000)
    ap.add_argument('--target-tests', type=int, default=200)
    ap.add_argument('--mode', choices=['g5-only', 'compare'], default='g5-only', help='Run only Group5, or also compare to Group2 and Group3')
    ap.add_argument('--per-config', action='store_true', help='Instead of interleaving across all configs, iterate variants within each config before moving on')
    ap.add_argument('--out', type=str, help='Optional output path (auto-set by mode if omitted)')
    ap.add_argument('--sleep', type=float, default=0.0, help='Optional small sleep between runs to reduce resource contention')
    args = ap.parse_args()

    if not args.out:
        if args.mode == 'g5-only':
            args.out = str(ROOT / 'gardeners' / 'group5' / 'benchmarks' / 'results_g5.csv')
        else:
            args.out = str(ROOT / 'gardeners' / 'group5' / 'benchmarks' / 'results_compare.csv')

    configs = find_config_files()
    variants = variants_set()
    if args.per_config:
        plan: List[Tuple[Path, Variant]] = []
        while len(plan) < args.target_tests:
            for cfg in configs:
                for var in variants:
                    plan.append((cfg, var))
                    if len(plan) >= args.target_tests:
                        break
                if len(plan) >= args.target_tests:
                    break
    else:
        plan = plan_runs(configs, variants, args.target_tests)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == 'g5-only':
        header = [
            'config', 'variant', 'overrides', 'turns',
            'g5_growth', 'g5_plants', 'g5_time'
        ]
    else:
        header = [
            'config', 'variant', 'overrides', 'turns',
            'g5_growth', 'g5_plants', 'g5_time',
            'g2_growth', 'g2_plants', 'g2_time',
            'g3_growth', 'g3_plants', 'g3_time',
            'delta_vs_g2', 'delta_vs_g3', 'ratio_vs_g2', 'ratio_vs_g3'
        ]

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for i, (cfg, var) in enumerate(plan, 1):
            prev = apply_overrides(var.overrides)
            try:
                res_g5 = run_one(cfg, args.turns, Gardener5)
                if args.mode == 'compare':
                    res_g2 = run_one(cfg, args.turns, Gardener2)
                    res_g3 = run_one(cfg, args.turns, Gardener3)
                else:
                    res_g2 = res_g3 = None
            finally:
                restore_overrides(prev)

            g5g = res_g5['final_growth']
            row: Dict[str, Any] = {
                'config': str(cfg),
                'variant': var.name,
                'overrides': json.dumps(var.overrides),
                'turns': args.turns,
                'g5_growth': g5g,
                'g5_plants': res_g5['plants_placed'],
                'g5_time': res_g5['placement_time'],
            }
            if res_g2 and res_g3:
                g2g = res_g2['final_growth']; g3g = res_g3['final_growth']
                row.update({
                    'g2_growth': g2g,
                    'g2_plants': res_g2['plants_placed'],
                    'g2_time': res_g2['placement_time'],
                    'g3_growth': g3g,
                    'g3_plants': res_g3['plants_placed'],
                    'g3_time': res_g3['placement_time'],
                    'delta_vs_g2': g5g - g2g,
                    'delta_vs_g3': g5g - g3g,
                    'ratio_vs_g2': (g5g / g2g) if g2g else 0.0,
                    'ratio_vs_g3': (g5g / g3g) if g3g else 0.0,
                })
            writer.writerow(row)
            f.flush()

            if res_g2 and res_g3:
                print(
                    f"[{i}/{len(plan)}] {cfg.name} | {var.name} -> "
                    f"G5={g5g:.2f} (plants={res_g5['plants_placed']}, time={res_g5['placement_time']:.2f}s) | "
                    f"G2={res_g2['final_growth']:.2f} | G3={res_g3['final_growth']:.2f} | dG2={row['delta_vs_g2']:.2f} dG3={row['delta_vs_g3']:.2f}"
                )
            else:
                print(
                    f"[{i}/{len(plan)}] {cfg.name} | {var.name} -> "
                    f"G5={g5g:.2f} (plants={res_g5['plants_placed']}, time={res_g5['placement_time']:.2f}s)"
                )
            if args.sleep:
                time.sleep(args.sleep)
