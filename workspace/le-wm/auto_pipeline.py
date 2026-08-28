"""End-to-end watcher: setup → launch pipeline → poll → pull → release.

Polls /workspace/setup_done on AutoDL. When EXIT=0 appears:
1. Verifies torch + Push-T data.
2. Launches run_full.sh in nohup background.
3. Polls /workspace/run_full_done.
4. Pulls all results to /workspace/lewm_autodl_results_v2/.
5. Releases the instance.

If setup or pipeline failed (EXIT != 0) it stops and prints diagnostics —
does NOT release the instance, so the user can SSH in to debug.
"""
import json
import os
import sys
import time
from pathlib import Path

import paramiko
import requests


TOKEN = open('/root/.claude/skills/autoDL/.env').read().split('=', 1)[1].strip()
# Persistent location (survives WSL restart). Fall back to /tmp for backward-compat.
_ssh_p = Path('/workspace/autodl_ssh.json')
if not _ssh_p.exists():
    _ssh_p = Path('/tmp/autodl_ssh.json')
SSH_INFO = json.loads(_ssh_p.read_text())
HOST = SSH_INFO['host']; PORT = SSH_INFO['port']
PASS = SSH_INFO['password']; UUID = SSH_INFO['uuid']

OUT_DIR = Path('/workspace/lewm_autodl_results_v2')


def ssh_open():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username='root', password=PASS, timeout=60)
    return c


def remote_exec(client, cmd, timeout=30):
    """Returns (exit, stdout). For short commands."""
    chan = client.get_transport().open_session()
    chan.set_combine_stderr(True); chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b''
    try:
        while True:
            d = chan.recv(8192)
            if not d: break
            out += d
    except Exception: pass
    return chan.recv_exit_status(), out.decode('utf-8','replace')


def wait_marker(remote_path, label, sleep_s=120, max_loops=240):
    """Block until `cat <remote_path>` returns 'EXIT=...'. Returns the value."""
    print(f"\n=== waiting on {label} ({remote_path}) ===")
    for i in range(max_loops):
        try:
            c = ssh_open()
            _, out = remote_exec(c, f'cat {remote_path} 2>/dev/null', timeout=15)
            c.close()
            out = out.strip()
            if out.startswith('EXIT='):
                print(f"\n>>> {label} finished with {out}")
                return out
        except Exception as e:
            print(f"  [{label}] poll error: {type(e).__name__}: {e}")
        # progress check
        try:
            c = ssh_open()
            log = remote_path.replace('_done', '.log')
            _, info = remote_exec(c, f'wc -l {log} 2>/dev/null; tail -1 {log} 2>/dev/null', timeout=10)
            c.close()
            print(f"  [{time.strftime('%H:%M:%S')}] {info.strip()[:200]}")
        except Exception: pass
        time.sleep(sleep_s)
    raise RuntimeError(f"{label} timed out after {max_loops*sleep_s/60:.1f} min")


def verify_setup():
    print("\n=== verifying setup ===")
    c = ssh_open()
    PY = '/root/miniconda3/envs/lewm310/bin/python'
    _, out = remote_exec(c, f'{PY} -c "import torch, einops, sklearn, stable_pretraining, stable_worldmodel; print(\\"OK\\", torch.__version__, torch.cuda.is_available())"; ls -la /workspace/stablewm_home/*.h5 2>&1 | head -1')
    print(out)
    c.close()
    return 'OK 2.6.0' in out and '.h5' in out and 'No such' not in out


def launch_pipeline(steps=40000, probe_every=500, batch=32):
    """Launches run_full.sh ONLY if no pipeline is already running.

    Lesson learned 2026-05-10: blindly relaunching while an old run_full.sh
    is still alive caused two bash scripts to compete on the same output paths.
    The callbacks' init-time `unlink()` then nuked the old process's data files.
    Fix: detect existing pipeline before launching. If found, attach (just poll)
    instead of relaunching.
    """
    print(f"\n=== launch_pipeline check ===")
    c = ssh_open()
    _, existing_run = remote_exec(c, 'pgrep -af run_full.sh', timeout=10)
    _, existing_train = remote_exec(c, 'pgrep -af "train.py.*output_model_name"', timeout=10)
    if existing_run.strip() or existing_train.strip():
        print(f"!! pipeline already running on remote — NOT relaunching. Will poll only.")
        print(f"   run_full.sh: {existing_run.strip()[:200]}")
        print(f"   train.py:    {existing_train.strip()[:200]}")
        c.close()
        return  # do nothing — just attach via wait_marker

    print(f"=== launching run_full.sh {steps} {probe_every} {batch} ===")
    cmd = (
        "cd /workspace/le-wm && "
        "rm -f /workspace/run_full.log /workspace/run_full_done && "
        f"nohup bash -c \"./run_full.sh {steps} {probe_every} {batch} > /workspace/run_full.log 2>&1; "
        "echo EXIT=\\$? > /workspace/run_full_done\" >/dev/null 2>&1 &"
    )
    chan = c.get_transport().open_session()
    chan.exec_command(cmd)
    chan.recv_exit_status()
    time.sleep(5)
    _, out = remote_exec(c, 'pgrep -af run_full | head -3', timeout=10)
    print(out)
    c.close()


def pull_all():
    print(f"\n=== pulling all results to {OUT_DIR} ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = ssh_open()
    sftp = c.open_sftp()

    def walk(rdir, ldir):
        Path(ldir).mkdir(parents=True, exist_ok=True)
        try:
            for entry in sftp.listdir_attr(rdir):
                rp = f"{rdir}/{entry.filename}"; lp = f"{ldir}/{entry.filename}"
                if entry.st_mode and (entry.st_mode >> 14) & 1:
                    walk(rp, lp)
                else:
                    sftp.get(rp, lp)
        except IOError as e:
            print(f"  skip {rdir}: {e}")

    print("  pulling aggregated/")
    walk('/workspace/le-wm/aggregated', f'{OUT_DIR}/aggregated')
    for variant in ['baseline', 'uvlowr_r4', 'randdiff']:
        for seed in [0, 1, 2]:
            run = f'/workspace/stablewm_home/{variant}_seed{seed}'
            local = f'{OUT_DIR}/{variant}_seed{seed}'
            Path(local).mkdir(parents=True, exist_ok=True)
            for fn in ['eval.json', 'eval_rollout_per_step.csv',
                       'engineering_metrics.json', 'final_analyses.json',
                       'jacobian_probe.jsonl', 'probing.jsonl', 'latent_cov.jsonl',
                       'config.yaml']:
                try:
                    sftp.get(f'{run}/{fn}', f'{local}/{fn}')
                except IOError:
                    pass
            try:
                Path(f'{local}/csv/version_0').mkdir(parents=True, exist_ok=True)
                sftp.get(f'{run}/csv/version_0/metrics.csv',
                         f'{local}/csv/version_0/metrics.csv')
            except IOError: pass
            print(f"  pulled {variant}_seed{seed}")
    # Also pull /workspace/run_full.log for diagnostics
    try:
        sftp.get('/workspace/run_full.log', f'{OUT_DIR}/run_full.log')
    except IOError: pass
    sftp.close(); c.close()


def release():
    print(f"\n=== releasing {UUID} ===")
    r = requests.post('https://api.autodl.com/api/v1/dev/instance/pro/power_off',
                      headers={'Authorization': TOKEN},
                      json={'instance_uuid': UUID}, timeout=30)
    print('power_off:', r.json())
    # Wait shutdown
    for _ in range(30):
        time.sleep(10)
        r = requests.get('https://api.autodl.com/api/v1/dev/instance/pro/status',
                         headers={'Authorization': TOKEN},
                         params={'instance_uuid': UUID}, timeout=15)
        s = r.json().get('data')
        print(f'  status: {s}')
        if s == 'shutdown': break
    r = requests.post('https://api.autodl.com/api/v1/dev/instance/pro/release',
                      headers={'Authorization': TOKEN},
                      json={'instance_uuid': UUID}, timeout=30)
    print('release:', r.json())
    # Balance
    r = requests.post('https://api.autodl.com/api/v1/dev/wallet/balance',
                      headers={'Authorization': TOKEN}, timeout=15)
    b = r.json()['data']['assets'] / 1000
    print(f'\nBalance after release: ¥{b:.2f}')


def main():
    # 1. Wait for setup
    setup_status = wait_marker('/workspace/setup_done', 'setup', sleep_s=120, max_loops=120)
    if not setup_status.startswith('EXIT=0'):
        print(f"!! setup failed: {setup_status}; not auto-launching pipeline")
        return 1

    if not verify_setup():
        print("!! verify_setup failed; not launching")
        return 2

    # 2. Launch pipeline
    launch_pipeline(steps=40000, probe_every=500, batch=32)

    # 3. Wait for pipeline
    pipeline_status = wait_marker('/workspace/run_full_done', 'pipeline',
                                  sleep_s=300, max_loops=120)

    # 4. Pull results
    pull_all()

    # 5. Release
    if pipeline_status.startswith('EXIT=0'):
        release()
    else:
        print(f"!! pipeline did not end cleanly: {pipeline_status}; NOT auto-releasing.")

    print('\n=== auto_pipeline DONE ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
