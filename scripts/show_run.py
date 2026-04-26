#!/usr/bin/env python3
"""Show a Stage 3 run's test metrics + val trajectory in a compact format.

Usage:
    python scripts/show_run.py <output_dir>
    python scripts/show_run.py models/stage3_risk_deberta_v3_run14_hardonly
    python scripts/show_run.py models/stage3_risk_deberta_v3_*  # via shell glob
"""

import json
import sys
from pathlib import Path


def show(output_dir: Path) -> None:
    name = output_dir.name
    metrics_path = output_dir / "test_metrics.json"
    if not metrics_path.exists():
        print(f"{name}: NO test_metrics.json")
        return

    m = json.loads(metrics_path.read_text())
    o = m["overall"]
    pc = o["per_class"]
    hard_only = m["our_model_hard_only_macro_f1"]

    # Find latest checkpoint with trainer_state.json
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    best_val = None
    val_history = []
    for c in reversed(ckpts):
        ts = c / "trainer_state.json"
        if ts.exists():
            s = json.loads(ts.read_text())
            best_val = s.get("best_metric")
            val_history = [
                e for e in s.get("log_history", []) if "eval_val_macro_f1" in e
            ]
            break

    print(f"=== {name} ===")
    print(
        f"Test n={o['n']}: macro_f1={o['macro_f1']:.4f}  acc={o['accuracy']:.4f}  "
        f"hard-only={hard_only:.4f}"
    )
    for c in ("LOW", "MEDIUM", "HIGH"):
        print(
            f"  {c:<7} P={pc[c]['precision']:.3f}  R={pc[c]['recall']:.3f}  "
            f"F1={pc[c]['f1']:.3f}"
        )
    if best_val is not None:
        print(f"best_val: {best_val:.4f}")
    if val_history:
        print("Val trajectory:")
        for e in val_history:
            print(
                f"  ep{e['epoch']:.0f}: macro={e['eval_val_macro_f1']:.4f}  "
                f"loss={e['eval_loss']:.4f}  "
                f"L={e['eval_val_f1_LOW']:.3f}/M={e['eval_val_f1_MEDIUM']:.3f}/"
                f"H={e['eval_val_f1_HIGH']:.3f}"
            )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists():
            print(f"{path}: not found", file=sys.stderr)
            continue
        show(p)
        print()


if __name__ == "__main__":
    main()
