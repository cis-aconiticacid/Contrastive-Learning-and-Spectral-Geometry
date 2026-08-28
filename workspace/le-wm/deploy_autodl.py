"""Deploy the lewm low-rank pipeline to an AutoDL Pro instance.

Usage:
    python deploy_autodl.py setup     # SCP files + run remote_setup.sh
    python deploy_autodl.py run       # launch run_full.sh in tmux on the instance
    python deploy_autodl.py status    # check tmux + recent log output
    python deploy_autodl.py pull      # pull aggregated/ + key per-seed CSV/JSON back
    python deploy_autodl.py release   # power_off + release + delete keypair
"""
import os
import sys
import time
import json
import requests
from pathlib import Path

import paramiko


HERE = Path(__file__).parent
TOKEN = open('/root/.claude/skills/autoDL/.env').read().split('=', 1)[1].strip()
INSTANCE_UUID = os.environ.get('AUTODL_INSTANCE_UUID', 'pro-778340595241')


def get_snapshot():
    r = requests.get('https://api.autodl.com/api/v1/dev/instance/pro/snapshot',
                     headers={'Authorization': TOKEN},
                     params={'instance_uuid': INSTANCE_UUID}, timeout=30)
    return r.json().get('data') or {}


def open_ssh():
    snap = get_snapshot()
    host = snap['proxy_host']
    port = snap['ssh_port']
    pwd = snap['root_password']
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username='root', password=pwd, timeout=60,
              banner_timeout=60, auth_timeout=60)
    return c, snap


def run_cmd(client, cmd, stream=True, env=None):
    """Run command, stream stdout. Returns exit_code."""
    full = cmd
    if env:
        prefix = " ".join(f"{k}={v}" for k, v in env.items())
        full = f"{prefix} bash -c {json.dumps(cmd)}"
    chan = client.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(full)
    while True:
        if chan.recv_ready():
            data = chan.recv(4096).decode('utf-8', errors='replace')
            if stream:
                sys.stdout.write(data)
                sys.stdout.flush()
        if chan.exit_status_ready():
            # Drain
            while chan.recv_ready():
                data = chan.recv(4096).decode('utf-8', errors='replace')
                if stream:
                    sys.stdout.write(data); sys.stdout.flush()
            return chan.recv_exit_status()
        time.sleep(0.05)


def cmd_setup():
    """Upload patched files and run remote_setup.sh."""
    print("=== Connecting ===")
    client, snap = open_ssh()
    print(f"connected to {snap['proxy_host']}:{snap['ssh_port']}")

    # First: clone le-wm fresh on the instance (network_turbo lets us reach github)
    print("\n=== Cloning le-wm on instance ===")
    code = run_cmd(client, (
        "source /etc/network_turbo 2>/dev/null; "
        "mkdir -p /workspace && cd /workspace && "
        "if [ ! -d le-wm/.git ]; then "
        "  rm -rf le-wm && "
        "  git clone --depth 1 https://github.com/lucas-maes/le-wm.git; "
        "fi && ls /workspace/le-wm/ | head -5"
    ))
    if code != 0:
        print(f"[setup] clone failed (exit={code})")
        client.close()
        return False

    # SFTP upload (overwrites the just-cloned files with our patched versions)
    print("\n=== Uploading patched files via SFTP ===")
    sftp = client.open_sftp()
    try:
        sftp.mkdir('/workspace')
    except IOError:
        pass
    try:
        sftp.mkdir('/workspace/le-wm')
    except IOError:
        pass

    files = [
        'module.py',
        'train.py',
        'callbacks.py',
        'eval_rollout.py',
        'aggregate.py',
        'jacobian_probe.py',
        'final_analyses.py',
        'spt_patch.py',
        'run_full.sh',
        'remote_setup.sh',
    ]
    for f in files:
        local = HERE / f
        remote = f'/workspace/le-wm/{f}'
        sftp.put(str(local), remote)
        print(f"  uploaded {f}")
    # Make scripts executable
    sftp.chmod('/workspace/le-wm/run_full.sh', 0o755)
    sftp.chmod('/workspace/le-wm/remote_setup.sh', 0o755)
    sftp.close()

    print("\n=== Launching remote_setup.sh via nohup (SSH-disconnect-safe) ===")
    run_cmd(client, "rm -f /workspace/setup_done /workspace/setup.log; "
                    "pkill -9 -f remote_setup 2>/dev/null; sleep 1")
    # nohup in background; redirect output. Wrap in (...)& to fully detach.
    cmd = (
        "cd /workspace/le-wm && "
        "nohup bash -c 'bash remote_setup.sh > /workspace/setup.log 2>&1; "
        "echo EXIT=$? > /workspace/setup_done' >/dev/null 2>&1 &"
    )
    run_cmd(client, cmd)
    run_cmd(client, "sleep 1; pgrep -af remote_setup")
    client.close()

    # Poll
    print("\n=== Polling setup progress ===")
    while True:
        try:
            client, _ = open_ssh()
            r1 = run_cmd_capture(client, "test -f /workspace/setup_done && cat /workspace/setup_done")
            if r1.strip().startswith("EXIT="):
                print(f"\nSETUP FINISHED: {r1.strip()}")
                # Show last lines
                print("=== last setup log lines ===")
                run_cmd(client, "tail -30 /workspace/setup.log 2>&1")
                client.close()
                return r1.strip().startswith("EXIT=0")
            log_size = run_cmd_capture(client, "stat -c%s /workspace/setup.log 2>/dev/null || echo 0")
            log_tail = run_cmd_capture(client, "tail -1 /workspace/setup.log 2>/dev/null | head -c 120")
            print(f"  [poll] log={log_size.strip()}B | last: {log_tail.strip()}")
            client.close()
        except Exception as e:
            print(f"  [poll] {type(e).__name__}: {e}")
        time.sleep(60)


def run_cmd_capture(client, cmd, timeout=30):
    """Run command, return stdout as string."""
    chan = client.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(cmd)
    chan.settimeout(timeout)
    out = b""
    try:
        while True:
            data = chan.recv(4096)
            if not data:
                break
            out += data
    except Exception:
        pass
    return out.decode("utf-8", errors="replace")


def cmd_run(steps=5000, probe=500, batch=32):
    """Launch run_full.sh in a tmux session so it survives SSH disconnect."""
    print("=== Launching training in tmux ===")
    client, _ = open_ssh()
    # Re-upload run_full.sh + scripts in case we tweaked them
    sftp = client.open_sftp()
    for f in ['run_full.sh', 'train.py', 'callbacks.py', 'eval_rollout.py',
              'aggregate.py', 'module.py']:
        sftp.put(str(HERE / f), f'/workspace/le-wm/{f}')
    sftp.chmod('/workspace/le-wm/run_full.sh', 0o755)
    sftp.close()

    # Use tmux for backgrounding
    tmux_cmd = (
        f"cd /workspace/le-wm && "
        f"export PATH=$HOME/.local/bin:$PATH && "
        f"export STABLEWM_HOME=/workspace/stablewm_home && "
        f"export WANDB_MODE=offline && "
        f"export SDL_VIDEODRIVER=dummy && "
        f"./run_full.sh {steps} {probe} {batch}"
    )
    # kill any existing run
    run_cmd(client, "tmux kill-session -t lewm 2>/dev/null || true", stream=False)
    cmd = f"tmux new-session -d -s lewm 'bash -lc {json.dumps(tmux_cmd + '; echo EXIT_CODE=$?; sleep 3600')}'"
    print(f"  cmd: {cmd[:200]}...")
    code = run_cmd(client, cmd)
    # Verify started
    run_cmd(client, "tmux ls; ls /workspace/le-wm/")
    client.close()
    print(f"\n[run] tmux launched (exit={code}). Use `python deploy_autodl.py status` to check.")


def cmd_status(tail=80):
    """Check tmux + recent training output."""
    client, _ = open_ssh()
    print("=== tmux sessions ===")
    run_cmd(client, "tmux ls 2>&1 || echo 'no tmux'")
    print("\n=== last training output ===")
    run_cmd(client, f"tmux capture-pane -t lewm -p -S -{tail} 2>&1 || echo 'no session'")
    print("\n=== output files in stablewm_home ===")
    run_cmd(client, "ls /workspace/stablewm_home/ 2>/dev/null | tail -30")
    print("\n=== eval status (look for eval.json files) ===")
    run_cmd(client, "find /workspace/stablewm_home -name 'eval.json' 2>/dev/null")
    print("\n=== aggregated status ===")
    run_cmd(client, "find /workspace/le-wm/aggregated -type f 2>/dev/null | head -40")
    client.close()


def cmd_pull(dest='/workspace/lewm_autodl_results'):
    """Download aggregated/ + per-seed metrics CSV+JSONs back to local."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    client, _ = open_ssh()
    sftp = client.open_sftp()

    # Walk + download key dirs
    def walk(remote_dir, local_dir):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        try:
            for entry in sftp.listdir_attr(remote_dir):
                rpath = f"{remote_dir}/{entry.filename}"
                lpath = f"{local_dir}/{entry.filename}"
                if entry.st_mode and (entry.st_mode >> 14) & 1:
                    # is dir
                    walk(rpath, lpath)
                else:
                    sftp.get(rpath, lpath)
        except IOError as e:
            print(f"  skip {remote_dir}: {e}")

    print(f"=== Pulling aggregated/ -> {dest}/aggregated/ ===")
    walk('/workspace/le-wm/aggregated', f"{dest}/aggregated")
    print(f"\n=== Pulling per-seed key files ===")
    for variant in ['baseline', 'uvlowr_r4', 'randdiff']:
        for seed in [0, 1, 2]:
            run_dir = f"/workspace/stablewm_home/{variant}_seed{seed}"
            local_run = f"{dest}/{variant}_seed{seed}"
            Path(local_run).mkdir(parents=True, exist_ok=True)
            for fn in ['jacobian_probe.jsonl', 'engineering_metrics.json',
                       'eval.json', 'eval_rollout_per_step.csv', 'config.yaml']:
                try:
                    sftp.get(f"{run_dir}/{fn}", f"{local_run}/{fn}")
                    print(f"  pulled {variant}_seed{seed}/{fn}")
                except IOError:
                    pass
            # CSV log
            try:
                Path(f"{local_run}/csv").mkdir(exist_ok=True)
                Path(f"{local_run}/csv/version_0").mkdir(exist_ok=True)
                sftp.get(f"{run_dir}/csv/version_0/metrics.csv",
                         f"{local_run}/csv/version_0/metrics.csv")
                print(f"  pulled {variant}_seed{seed}/csv/metrics.csv")
            except IOError:
                pass

    sftp.close()
    client.close()
    print(f"\n=== DONE: pulled to {dest}/ ===")


def cmd_release():
    print(f"=== Releasing instance {INSTANCE_UUID} ===")
    r = requests.post('https://api.autodl.com/api/v1/dev/instance/pro/power_off',
                      headers={'Authorization': TOKEN},
                      json={'instance_uuid': INSTANCE_UUID}, timeout=30)
    print('power_off:', r.json())
    time.sleep(5)
    r = requests.post('https://api.autodl.com/api/v1/dev/instance/pro/release',
                      headers={'Authorization': TOKEN},
                      json={'instance_uuid': INSTANCE_UUID}, timeout=30)
    print('release:', r.json())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    args = sys.argv[2:]
    if cmd == 'setup':
        cmd_setup()
    elif cmd == 'run':
        kwargs = {}
        for a in args:
            k, v = a.split('=', 1)
            kwargs[k] = int(v)
        cmd_run(**kwargs)
    elif cmd == 'status':
        cmd_status()
    elif cmd == 'pull':
        cmd_pull(*args)
    elif cmd == 'release':
        cmd_release()
    else:
        print("usage: deploy_autodl.py {setup|run|status|pull|release}")
        sys.exit(1)
