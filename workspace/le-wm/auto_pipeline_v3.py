"""v3 (frozen-encoder) watcher: setup → launch pipeline_v3 → poll → pull → release.

Same shape as auto_pipeline.py but:
- launches run_pipeline_v3.sh (does Stage-1 + M0 sanity + M1 frozen)
- pulls frozen_* variants
- pgrep precheck (lesson learned from 2026-05-10 watcher-relaunch race)
"""
import json
import os
import sys
import time
from pathlib import Path

import paramiko
import requests


TOKEN = open('/root/.claude/skills/autoDL/.env').read().split('=', 1)[1].strip()
_ssh_p = Path('/workspace/autodl_ssh.json')
if not _ssh_p.exists():
    _ssh_p = Path('/tmp/autodl_ssh.json')
SSH_INFO = json.loads(_ssh_p.read_text())
HOST = SSH_INFO['host']; PORT = SSH_INFO['port']
PASS = SSH_INFO['password']; UUID = SSH_INFO['uuid']

OUT_DIR = Path('/workspace/lewm_autodl_results_v3')


def ssh_open():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username='root', password=PASS, timeout=60,
              banner_timeout=120, auth_timeout=120)
    return c


def remote_exec(client, cmd, timeout=30):
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


def launch_pipeline(stage1_steps=20000, stage2_steps=8000, probe_every=500, batch=32):
    """Launch run_pipeline_v3.sh ONLY if no pipeline already running (pgrep precheck)."""
    print(f"\n=== launch_pipeline_v3 check ===")
    c = ssh_open()
    _, existing_run = remote_exec(c, 'pgrep -af run_pipeline_v3', timeout=10)
    _, existing_train = remote_exec(c, 'pgrep -af "train.py.*output_model_name"', timeout=10)
    if existing_run.strip() or existing_train.strip():
        print(f"!! pipeline already running on remote — NOT relaunching. Will poll only.")
        print(f"   run_pipeline_v3: {existing_run.strip()[:200]}")
        print(f"   train.py:       {existing_train.strip()[:200]}")
        c.close()
        return

    print(f"=== launching run_pipeline_v3.sh {stage1_steps} {stage2_steps} {probe_every} {batch} ===")
    cmd = (
        "cd /workspace/le-wm && "
        "rm -f /workspace/run_pipeline_v3.log /workspace/run_pipeline_v3_done && "
        f"nohup bash -c \"./run_pipeline_v3.sh {stage1_steps} {stage2_steps} {probe_every} {batch} > /workspace/run_pipeline_v3.log 2>&1; "
        "echo EXIT=\\$? > /workspace/run_pipeline_v3_done\" >/dev/null 2>&1 &"
    )
    chan = c.get_transport().open_session()
    chan.exec_command(cmd)
    chan.recv_exit_status()
    time.sleep(5)
    _, out = remote_exec(c, 'pgrep -af run_pipeline_v3 | head -3', timeout=10)
    print(out)
    c.close()


def pull_all():
    print(f"\n=== pulling all v3 results to {OUT_DIR} ===")
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

    print("  pulling aggregated_frozen/")
    walk('/workspace/le-wm/aggregated_frozen', f'{OUT_DIR}/aggregated_frozen')

    # Pull stage1 + sanity + main variants
    variant_seeds = [
        ('stage1_baseline', [0]),
        ('frozen_sanity', [0]),
        ('frozen_baseline', [0, 1]),
        ('frozen_uvlowr_r4', [0, 1]),
        ('frozen_randdiff', [0, 1]),
    ]
    for variant, seeds in variant_seeds:
        for seed in seeds:
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

    # Also pull Stage-1 ckpt for future reuse + log
    try:
        sftp.get('/workspace/stablewm_home/stage1_baseline_seed0/stage1_baseline_seed0_weights.ckpt',
                 f'{OUT_DIR}/stage1_baseline_seed0_weights.ckpt')
        print("  pulled Stage-1 weights ckpt")
    except IOError as e:
        print(f"  skip stage1 weights: {e}")

    try:
        sftp.get('/workspace/run_pipeline_v3.log', f'{OUT_DIR}/run_pipeline_v3.log')
    except IOError: pass
    sftp.close(); c.close()


def release():
    print(f"\n=== releasing {UUID} ===")
    r = requests.post('https://api.autodl.com/api/v1/dev/instance/pro/power_off',
                      headers={'Authorization': TOKEN},
                      json={'instance_uuid': UUID}, timeout=30)
    print('power_off:', r.json())
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
    r = requests.post('https://api.autodl.com/api/v1/dev/wallet/balance',
                      headers={'Authorization': TOKEN}, timeout=15)
    b = r.json()['data']['assets'] / 1000
    print(f'\nBalance after release: ¥{b:.2f}')


def main():
    setup_status = wait_marker('/workspace/setup_done', 'setup', sleep_s=120, max_loops=120)
    if not setup_status.startswith('EXIT=0'):
        print(f"!! setup failed: {setup_status}; not auto-launching pipeline")
        return 1

    if not verify_setup():
        print("!! verify_setup failed; not launching")
        return 2

    launch_pipeline(stage1_steps=20000, stage2_steps=8000, probe_every=500, batch=32)

    pipeline_status = wait_marker('/workspace/run_pipeline_v3_done', 'pipeline_v3',
                                  sleep_s=300, max_loops=144)  # up to 12h

    pull_all()

    if pipeline_status.startswith('EXIT=0'):
        release()
    else:
        print(f"!! pipeline did not end cleanly: {pipeline_status}; NOT auto-releasing.")
        print("   SSH in to diagnose. Run `python auto_pipeline_v3.py release` to release manually.")

    print('\n=== auto_pipeline_v3 DONE ===')
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'release':
        release()
        sys.exit(0)
    sys.exit(main())
