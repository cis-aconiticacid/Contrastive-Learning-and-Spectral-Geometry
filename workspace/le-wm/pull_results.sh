#!/bin/bash
# Pull AutoDL v5 results back to local destination.
# Run this AFTER AutoDL subset completes (check via monitor_v5.py).
#
# Usage:
#   ./pull_results.sh <autodl_ssh> <local_dest>
#
# Example:
#   ./pull_results.sh root@region-N.seetacloud.com -p 12345 ./lewm_autodl_results_v5/
#
# Pulls:
#   - all v5_baseline_40K, v5_uvlowr_40K (B4' AutoDL portion)
#   - all v5_pm_* (B9 PointMaze)
#   - all jsonl/json analysis outputs
set -e

AUTODL_SSH=${1:?"Usage: $0 <user@host or alias> <local_dest_dir>"}
LOCAL_DEST=${2:-./lewm_autodl_results_v5/}

mkdir -p "$LOCAL_DEST"

echo "==== Pulling B4' AutoDL outputs ===="
for run in v5_baseline_40K v5_uvlowr_40K; do
    scp -r "${AUTODL_SSH}:/workspace/stablewm_home/${run}" "${LOCAL_DEST}/" 2>&1 | tail -2 || \
        echo "WARN: ${run} not pulled"
done

echo "==== Pulling B9 PointMaze outputs ===="
scp -r "${AUTODL_SSH}:/workspace/stablewm_home/v5_pm_*" "${LOCAL_DEST}/" 2>&1 | tail -2 || \
    echo "WARN: B9 not pulled"

echo "==== Pulling logs ===="
mkdir -p "${LOCAL_DEST}/logs"
scp -r "${AUTODL_SSH}:/workspace/le-wm/logs/*.log" "${LOCAL_DEST}/logs/" 2>&1 | tail -2 || \
    echo "WARN: logs not pulled"

echo "==== Done ===="
echo "Local dest size:"
du -sh "${LOCAL_DEST}"

echo ""
echo "Next step: release the AutoDL instance via the autoDL skill"
echo "  power_off → wait shutdown → release"
