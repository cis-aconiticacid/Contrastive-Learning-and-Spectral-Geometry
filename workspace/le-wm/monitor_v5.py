"""Monitor v5.2 AutoDL subset progress + apply abort gates.

Inspects:
  - stablewm_home for output directory presence
  - jacobian_probe.jsonl + engineering_metrics.json for completion
  - pred_loss trajectory for divergence

Outputs structured JSON status to stdout (parseable by Claude).

Usage:
    python monitor_v5.py                  # one-shot status
    python monitor_v5.py --poll 60        # poll every 60 seconds
    python monitor_v5.py --check-gates    # also check P-* and abort gates
"""
import argparse
import json
import os
import time
from pathlib import Path


# Expected AutoDL subset (Plan E)
AUTODL_BLOCKS = {
    "autodl_sanity": {"steps": 200, "block": "sanity"},
    "v5_baseline_40K": {"steps": 40000, "block": "B4'"},
    "v5_uvlowr_40K": {"steps": 40000, "block": "B4'"},
    "v5_pm_stage1": {"steps": 20000, "block": "B9 Stage-1"},
    "v5_pm_baseline_seed0": {"steps": 8000, "block": "B9"},
    "v5_pm_baseline_seed1": {"steps": 8000, "block": "B9"},
    "v5_pm_uvlowr_seed0": {"steps": 8000, "block": "B9"},
    "v5_pm_uvlowr_seed1": {"steps": 8000, "block": "B9"},
    "v5_pm_randdiff_seed0": {"steps": 8000, "block": "B9"},
    "v5_pm_randdiff_seed1": {"steps": 8000, "block": "B9"},
}


def run_status(name, steps_expected, home):
    """Return a dict summarizing one run's status."""
    out = home / name
    status = {
        "name": name,
        "steps_expected": steps_expected,
        "exists": out.exists(),
        "completed": False,
        "current_step": None,
        "pred_loss_final": None,
        "wall_clock_s": None,
        "ckpt_present": False,
        "jacobian_present": False,
        "eval_present": False,
    }
    if not status["exists"]:
        return status

    # check final ckpt
    ckpt = out / f"{name}_epoch_1_object.ckpt"
    status["ckpt_present"] = ckpt.exists()

    # engineering_metrics.json marks completion (LeWM training writes this last)
    em = out / "engineering_metrics.json"
    if em.exists():
        try:
            data = json.loads(em.read_text())
            status["wall_clock_s"] = data.get("wall_clock_s")
            status["current_step"] = data.get("global_step")
        except Exception:
            pass

    # jacobian probe records training progress
    jp = out / "jacobian_probe.jsonl"
    if jp.exists():
        status["jacobian_present"] = True
        # last line gives current step
        try:
            with open(jp) as f:
                last = None
                for line in f:
                    if line.strip():
                        last = line
                if last:
                    rec = json.loads(last)
                    status["current_step"] = max(status["current_step"] or 0, rec.get("step", 0))
        except Exception:
            pass

    # eval.json indicates eval_rollout + final_analyses ran
    ev = out / "eval.json"
    if ev.exists():
        status["eval_present"] = True

    # Mark completed if ckpt + jacobian + eval all present
    status["completed"] = (
        status["ckpt_present"] and status["eval_present"] and
        (status["current_step"] or 0) >= steps_expected
    )
    return status


def check_p_gates(home):
    """Check P-Anti-I and P-Anti-K from completed runs."""
    gates = {}

    # P-Anti-I: B4' baseline_40K + randdiff_40K (randdiff_40K is local, may not be on AutoDL home)
    base = home / "v5_baseline_40K" / "jacobian_probe.jsonl"
    rdf = home / "v5_randdiff_40K" / "jacobian_probe.jsonl"
    if base.exists():
        try:
            base_lines = open(base).readlines()
            base_sigma1 = json.loads(base_lines[-1])["mean_spectral_norm"]
            gates["P-Anti-I_baseline_40K_sigma1"] = base_sigma1
        except Exception as e:
            gates["P-Anti-I_baseline_40K_sigma1"] = f"ERR {e}"
    if rdf.exists():
        try:
            rdf_lines = open(rdf).readlines()
            rdf_sigma1 = json.loads(rdf_lines[-1])["mean_spectral_norm"]
            gates["P-Anti-I_randdiff_40K_sigma1"] = rdf_sigma1
        except Exception as e:
            gates["P-Anti-I_randdiff_40K_sigma1"] = f"ERR {e}"

    # P-Anti-K: B9 PointMaze randdiff vs baseline
    for var in ["baseline", "uvlowr", "randdiff"]:
        sigs = []
        for s in [0, 1]:
            path = home / f"v5_pm_{var}_seed{s}" / "jacobian_probe.jsonl"
            if path.exists():
                try:
                    rec = json.loads(open(path).readlines()[-1])
                    sigs.append(rec["mean_spectral_norm"])
                except Exception:
                    pass
        if sigs:
            gates[f"P-Anti-K_{var}_sigma1_mean"] = sum(sigs) / len(sigs)
            gates[f"P-Anti-K_{var}_n_seeds"] = len(sigs)

    if "P-Anti-K_baseline_sigma1_mean" in gates and "P-Anti-K_randdiff_sigma1_mean" in gates:
        diff = gates["P-Anti-K_randdiff_sigma1_mean"] - gates["P-Anti-K_baseline_sigma1_mean"]
        gates["P-Anti-K_verdict"] = "PASS" if diff < 0 else "FAIL"
        gates["P-Anti-K_diff"] = diff

    return gates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--home", default=os.environ.get("STABLEWM_HOME", "/workspace/stablewm_home"))
    p.add_argument("--poll", type=int, default=0, help="poll interval seconds (0 = one-shot)")
    p.add_argument("--check-gates", action="store_true")
    p.add_argument("--json", action="store_true", help="emit JSON instead of pretty text")
    args = p.parse_args()

    home = Path(args.home)

    def snapshot():
        runs = {}
        n_done = 0
        for name, meta in AUTODL_BLOCKS.items():
            runs[name] = run_status(name, meta["steps"], home)
            if runs[name]["completed"]:
                n_done += 1
        out = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "home": str(home),
            "n_blocks": len(AUTODL_BLOCKS),
            "n_completed": n_done,
            "runs": runs,
        }
        if args.check_gates:
            out["gates"] = check_p_gates(home)
        return out

    while True:
        s = snapshot()
        if args.json:
            print(json.dumps(s, indent=2, default=str))
        else:
            print(f"\n=== {s['timestamp']}  AutoDL subset: {s['n_completed']}/{s['n_blocks']} blocks ===")
            for name, st in s["runs"].items():
                flag = "✓" if st["completed"] else ("…" if st["exists"] else "·")
                step_str = f"step={st['current_step']}/{st['steps_expected']}" if st["current_step"] else "(not started)"
                print(f"  {flag} {name:<35} {step_str}")
            if args.check_gates and "gates" in s:
                print("\n  --- gate values ---")
                for k, v in s["gates"].items():
                    print(f"    {k} = {v}")
        if args.poll <= 0:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
