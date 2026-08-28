# Resume guide — local computer restart while AutoDL pipeline running

**Status as of 2026-05-09 23:44**:
- AutoDL instance `pro-778340595241` (4090-48G, ¥3.03/hr) running 9-job pipeline
- Pipeline: baseline / uvlowr_r4 / randdiff × 3 seeds × 40K steps
- Progress at restart: 1/9 jobs done (baseline_seed0), uvlowr_r4_seed0 in progress
- ETA: ~8 more hours from 23:44 → ~07:30 next morning
- Pipeline is self-contained on AutoDL via nohup; survives local restart

## What WILL die on restart
- `auto_pipeline.py` watcher (does the auto-pull + auto-release at end)
- ScheduleWakeup loop (periodic status checks)
- This Claude Code session

## What survives
- AutoDL training (nohup'd on remote)
- All persisted code in `/workspace/le-wm/`, `/workspace/lewm_autodl_results/`
- AutoDL instance + its data
- Credentials at `/workspace/autodl_ssh.json`

## To resume after restart

Just tell Claude: "I'm back. Resume the AutoDL pipeline check."

Claude should then:
1. Read `/workspace/autodl_ssh.json` for SSH info
2. SSH in, check `cat /workspace/run_full_done` (EXIT=0 = pipeline done)
3. If done: `python /workspace/le-wm/auto_pipeline.py` (it'll skip setup wait, see pipeline done, pull + release)
   - OR manually pull + release via deploy_autodl.py
4. If still running: re-spawn watcher OR set up new ScheduleWakeup loop

## Critical files Claude needs to know about

- `/workspace/le-wm/` — patched LeWM codebase
- `/workspace/le-wm/auto_pipeline.py` — watcher, can be re-launched
- `/workspace/le-wm/deploy_autodl.py` — manual pull/release commands
- `/workspace/autodl_ssh.json` — SSH credentials
- `/workspace/lewm_autodl_results/` — old 5K-step results (already pulled)
- `/workspace/lewm_autodl_results_v2/` — target dir for new 40K results
- `/root/.claude/skills/autoDL/.env` — AutoDL API token

## Manual SSH (just in case)

```
ssh -p 35527 root@connect.westd.seetacloud.com
# password: TlSlqrAvDuSO
```

Inside: `tail /workspace/run_full.log`, `cat /workspace/run_full_done`,
`ls /workspace/stablewm_home/*_seed*/eval.json | wc -l`

## ⚠ AutoDL billing

Instance bills ¥3.03/hr while RUNNING. If pipeline crashes mid-way and we don't
release promptly, we'll bleed money. After restart, FIRST PRIORITY is to check
status and release if done.
