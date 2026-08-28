#!/bin/bash
# Runs ON the AutoDL instance. Sets up env, downloads Push-T, applies our patches.
# Patches are uploaded via SFTP before this script runs.
set -e

echo "=== STAGE 0: enable network_turbo (academic/github/HF proxy) ==="
[ -f /etc/network_turbo ] && source /etc/network_turbo
echo "  proxy: ${http_proxy:-none}"

echo "=== STAGE 1: create conda python-3.10 env (avoids uv's slow python download) ==="
CONDA=/root/miniconda3
PY310=$CONDA/envs/lewm310/bin/python
if [ ! -x "$PY310" ]; then
  $CONDA/bin/conda create -n lewm310 python=3.10 -y -q
fi
$PY310 --version
PIP="$PY310 -m pip install --quiet"

echo "=== STAGE 2: ensure /workspace/le-wm exists; clone done by deploy driver ==="
ls /workspace/le-wm/{module,train,callbacks}.py

echo "=== STAGE 3: install torch + deps into the conda env ==="
cd /workspace/le-wm
PY=$PY310
$PIP "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
$PIP "stable-worldmodel[train]" "datasets>=2.0,<4.0" matplotlib zstandard h5py einops scikit-learn shapely pygame pymunk opencv-python-headless
$PIP --force-reinstall --no-deps "nvidia-cudnn-cu12==9.5.1.17"

# Symlink for compatibility with our paths (run_full.sh and aggregate.py reference .venv)
ln -sfn $CONDA/envs/lewm310 /workspace/le-wm/.venv

$PY -c "import torch, einops; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), 'cudnn:', torch.backends.cudnn.version())"

echo "=== STAGE 4: patch spt single-optimizer bug ==="
SPT_PATH=$($PY -c "import stable_pretraining, os; print(os.path.dirname(stable_pretraining.__file__))")
$PY <<PYEOF
from pathlib import Path
p = Path("$SPT_PATH") / "module.py"
s = p.read_text()
old = "        optimizers = self.optimizers()\n        logging.info(f\"`self.optimizers() gave us {len(optimizers)} optimizers\")"
new = "        optimizers = self.optimizers()\n        if not isinstance(optimizers, (list, tuple)):\n            optimizers = [optimizers]\n        logging.info(f\"`self.optimizers() gave us {len(optimizers)} optimizers\")"
if new in s:
    print("already patched")
else:
    p.write_text(s.replace(old, new))
    print("patched OK")
PYEOF

echo "=== STAGE 5: apply our patches (module.py / train.py / callbacks.py / eval_rollout.py / aggregate.py / run_full.sh) ==="
# These files were uploaded via SFTP before this script ran:
ls -la /workspace/le-wm/{module,train,callbacks,eval_rollout,aggregate}.py /workspace/le-wm/run_full.sh

echo "=== STAGE 6: download Push-T (via hf-mirror, no proxy) ==="
export STABLEWM_HOME=/workspace/stablewm_home
mkdir -p $STABLEWM_HOME
if [ ! -f $STABLEWM_HOME/pusht_expert_train.h5 ]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  export HF_ENDPOINT=https://hf-mirror.com
  echo "  downloading h5.zst from hf-mirror.com..."
  $PY -c "
from huggingface_hub import hf_hub_download
import time, os
t0 = time.time()
fp = hf_hub_download(repo_id='quentinll/lewm-pusht', filename='pusht_expert_train.h5.zst',
                    repo_type='dataset', local_dir='/workspace/stablewm_home')
print(f'downloaded {os.path.getsize(fp)/1e9:.2f} GB in {time.time()-t0:.1f}s')
"
  echo "  decompressing..."
  $PY -c "
import zstandard as z, time, os
t0 = time.time()
with open('/workspace/stablewm_home/pusht_expert_train.h5.zst','rb') as fi, open('/workspace/stablewm_home/pusht_expert_train.h5','wb') as fo:
    z.ZstdDecompressor().copy_stream(fi, fo)
print(f'decompressed {os.path.getsize(\"/workspace/stablewm_home/pusht_expert_train.h5\")/1e9:.2f} GB in {time.time()-t0:.1f}s')
"
  rm $STABLEWM_HOME/pusht_expert_train.h5.zst
fi
ls -la $STABLEWM_HOME/

echo "=== READY: setup complete ==="
