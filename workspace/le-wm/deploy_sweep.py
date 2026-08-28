#!/usr/bin/env python3
"""Orchestrate the randdiff λ-sweep + uvlowr seed1 on a fresh AutoDL 4090.

Usage:
  AUTODL_INSTANCE_UUID=pro-XXX python3 deploy_sweep.py setup     # clone + install deps + Push-T
  AUTODL_INSTANCE_UUID=pro-XXX python3 deploy_sweep.py upload    # upload stage1 ckpt + sweep script
  AUTODL_INSTANCE_UUID=pro-XXX python3 deploy_sweep.py launch    # tmux launch run_sweep_lam.sh
  AUTODL_INSTANCE_UUID=pro-XXX python3 deploy_sweep.py status    # check tmux + tail
  AUTODL_INSTANCE_UUID=pro-XXX python3 deploy_sweep.py pull      # pull v5_*_40K* results
"""
import os, sys, json, time
from pathlib import Path

# Reuse the original module's open_ssh / run_cmd helpers.
sys.path.insert(0, str(Path(__file__).parent))
# Patch hardcoded INSTANCE_UUID before import.
os.environ.setdefault('AUTODL_INSTANCE_UUID', 'pro-77914cae5d27')

import deploy_autodl as da  # type: ignore

HERE = Path(__file__).parent
STAGE1_LOCAL = Path('/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt')
STAGE1_REMOTE = '/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt'


def cmd_upload():
    """Upload stage1 ckpt + run_sweep_lam.sh + the v5 helper if missing."""
    print(f"=== Upload stage1 ckpt ({STAGE1_LOCAL.stat().st_size/2**20:.1f} MB) ===")
    client, snap = da.open_ssh()
    da.run_cmd(client, f"mkdir -p {os.path.dirname(STAGE1_REMOTE)}")
    sftp = client.open_sftp()
    sftp.put(str(STAGE1_LOCAL), STAGE1_REMOTE)
    print(f"  uploaded -> {STAGE1_REMOTE}")
    # Also upload run_sweep_lam.sh + spectral_postprocess.py (not in cmd_setup's file list)
    for f in ['run_sweep_lam.sh', 'spectral_postprocess.py']:
        sftp.put(str(HERE / f), f'/workspace/le-wm/{f}')
        print(f"  uploaded {f}")
    sftp.chmod('/workspace/le-wm/run_sweep_lam.sh', 0o755)
    sftp.close()
    # Verify
    da.run_cmd(client, f"ls -la {STAGE1_REMOTE} /workspace/le-wm/run_sweep_lam.sh")
    client.close()


def cmd_launch():
    """tmux-launch run_sweep_lam.sh on the instance."""
    print("=== Launching run_sweep_lam.sh in tmux ===")
    client, snap = da.open_ssh()
    da.run_cmd(client, "tmux kill-session -t lewm_sweep 2>/dev/null || true")
    inner = (
        "cd /workspace/le-wm && "
        "export PATH=$HOME/.local/bin:$PATH && "
        "export STABLEWM_HOME=/workspace/stablewm_home && "
        "export WANDB_MODE=offline && "
        "export SDL_VIDEODRIVER=dummy && "
        "export HF_ENDPOINT=https://hf-mirror.com && "
        "./run_sweep_lam.sh 2>&1 | tee /workspace/sweep.log"
    )
    tmux = f"tmux new-session -d -s lewm_sweep 'bash -lc {json.dumps(inner + '; echo EXIT_CODE=$?; sleep 7200')}'"
    da.run_cmd(client, tmux)
    da.run_cmd(client, "tmux ls; ls /workspace/le-wm/run_sweep_lam.sh")
    client.close()


def cmd_status():
    print("=== tmux ===")
    client, _ = da.open_ssh()
    da.run_cmd(client, "tmux ls 2>&1 || echo 'no tmux'")
    print("\n=== last sweep log (last 60 lines) ===")
    da.run_cmd(client, "tmux capture-pane -t lewm_sweep -p -S -60 2>&1 || tail -60 /workspace/sweep.log 2>/dev/null || echo 'no log'")
    print("\n=== completed runs ===")
    da.run_cmd(client, "ls /workspace/stablewm_home/v5_*_40K* 2>/dev/null | head -20")
    da.run_cmd(client, "find /workspace/stablewm_home/v5_*_40K* -name 'final_analyses.json' 2>/dev/null")
    print("\n=== GPU ===")
    da.run_cmd(client, "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>&1 | head -2")
    client.close()


def cmd_pull(dest='/workspace/lewm_autodl_results_v5_sweep'):
    print(f"=== Pulling results to {dest}/ ===")
    Path(dest).mkdir(parents=True, exist_ok=True)
    client, _ = da.open_ssh()
    sftp = client.open_sftp()
    runs = ['v5_randdiff_lam001_40K', 'v5_randdiff_lam003_40K',
            'v5_randdiff_lam005_40K', 'v5_uvlowr_40K_seed1']
    for r in runs:
        local_run = Path(dest) / r
        local_run.mkdir(exist_ok=True)
        remote = f'/workspace/stablewm_home/{r}'
        for fn in ['final_analyses.json', 'eval.json', 'eval_rollout_per_step.csv',
                   'engineering_metrics.json', 'jacobian_probe.jsonl', 'config.yaml',
                   'probing.jsonl', 'latent_cov.jsonl']:
            try:
                sftp.get(f"{remote}/{fn}", str(local_run / fn))
                print(f"  pulled {r}/{fn}")
            except IOError:
                pass
        # Also try ckpt (large) — optional
        for fn in [f'{r}_epoch_1_object.ckpt', f'{r}_weights.ckpt']:
            try:
                sftp.get(f"{remote}/{fn}", str(local_run / fn))
                print(f"  pulled {r}/{fn}")
            except IOError:
                pass
    # Also grab the top-level sweep log
    try:
        sftp.get('/workspace/sweep.log', f'{dest}/sweep.log')
        print(f"  pulled sweep.log")
    except IOError:
        pass
    sftp.close()
    client.close()
    print(f"\n=== DONE: pulled to {dest}/ ===")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'setup':
        da.cmd_setup()
    elif cmd == 'upload':
        cmd_upload()
    elif cmd == 'launch':
        cmd_launch()
    elif cmd == 'status':
        cmd_status()
    elif cmd == 'pull':
        cmd_pull(*sys.argv[2:])
    elif cmd == 'release':
        da.cmd_release()
    else:
        print("usage: deploy_sweep.py {setup|upload|launch|status|pull|release}")
        sys.exit(1)
