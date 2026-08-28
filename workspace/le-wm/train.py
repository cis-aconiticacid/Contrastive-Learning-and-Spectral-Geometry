import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from callbacks import (
    JacobianProbeCallback,
    ProbingCallback,
    LatentCovCallback,
    EngineeringMetricsCallback,
    PreClipGradNormCallback,
    KeepFrozenInEvalCallback,
)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    rd_weight = float(cfg.loss.get("rand_diff", {}).get("weight", 0.0))
    rd_eps = float(cfg.loss.get("rand_diff", {}).get("eps", 0.05))
    rd_dirs = int(cfg.loss.get("rand_diff", {}).get("num_dirs", 2))
    skip_sigreg = bool(cfg.get("freeze", {}).get("skip_sigreg", False))

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    if skip_sigreg:
        # Frozen-encoder mode: SIGReg has no gradient consumer; skip to save compute.
        total = output["pred_loss"]
    else:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        total = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Optional Scarvelis & Solomon (2024) randomized one-sided finite-difference
    # nuclear-norm-of-Jacobian penalty on the predictor.
    if stage == "fit" and rd_weight > 0.0:
        ctx_emb_d = ctx_emb.detach()
        base = self.model.predict(ctx_emb_d, ctx_act)
        rd_total = 0.0
        B = ctx_emb_d.size(0)
        for _ in range(rd_dirs):
            v = torch.randn_like(ctx_emb_d)
            n = v.flatten(1).norm(dim=-1).clamp_min(1e-8).view(B, 1, 1)
            v = v / n
            perturbed = self.model.predict(ctx_emb_d + rd_eps * v, ctx_act)
            diff = (perturbed - base).flatten(1)
            rd_total = rd_total + diff.norm(dim=-1).mean() / rd_eps
        rd_total = rd_total / rd_dirs
        output["rand_diff_loss"] = rd_total.detach()
        total = total + rd_weight * rd_total

    output["loss"] = total

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    # ===== Frozen-encoder mode (Stage-2 of frozen-encoder ablation) =====
    # CODEX-REVIEW FIX: strict-load for safety; eval-mode enforced via callback
    # (not monkey-patch) so *_object.ckpt remains picklable.
    freeze_cfg = cfg.get("freeze", {})
    if freeze_cfg.get("enabled", False):
        ckpt_path = freeze_cfg["stage1_ckpt"]
        print(f"[FREEZE] Loading Stage-1 ckpt: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        def _load_into_strict(submodule, prefix, name):
            sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            assert len(sub_sd) > 0, (
                f"[FREEZE] FATAL: no ckpt keys found for prefix '{prefix}' "
                f"(name={name}). Available top-level prefixes: "
                f"{sorted(set(k.split('.')[0]+'.'+k.split('.')[1]+'.' for k in sd.keys() if '.' in k))[:8]}"
            )
            missing, unexp = submodule.load_state_dict(sub_sd, strict=True)
            assert not missing and not unexp, (
                f"[FREEZE] FATAL: {name} ckpt mismatch: "
                f"missing={missing[:5]} unexp={unexp[:5]}"
            )
            print(f"[FREEZE]   {name}: loaded {len(sub_sd)} keys strictly", flush=True)

        _load_into_strict(world_model.encoder,   "model.encoder.",   "encoder")
        _load_into_strict(world_model.projector, "model.projector.", "projector")
        _load_into_strict(world_model.pred_proj, "model.pred_proj.", "pred_proj")

        def _set_frozen(module, name):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
            n_params = sum(p.numel() for p in module.parameters())
            print(f"[FREEZE]   froze {name}: {n_params:,} params, "
                  f"all requires_grad=False (eval mode enforced by callback)", flush=True)

        _set_frozen(world_model.encoder,   "encoder")
        _set_frozen(world_model.projector, "projector")
        _set_frozen(world_model.pred_proj, "pred_proj")

        trainable = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in world_model.parameters())
        print(f"[FREEZE] trainable: {trainable:,} / total {total_params:,} "
              f"({100*trainable/total_params:.1f}%)", flush=True)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        # When wandb is disabled, fall back to a CSVLogger so per-step metrics
        # are still captured for later analysis.
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(run_dir), name="csv", version=0)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    # Probes: predictor Jacobian / Ridge probe (R²) / latent covariance — every
    # `probe_every` steps with step=0 control. Engineering once. Grad norm pre-clip.
    probe_every = int(cfg.get("probe_every", 500))
    jacobian_cb = JacobianProbeCallback(
        output_path=str(run_dir / "jacobian_probe.jsonl"),
        every_n_steps=probe_every,
        n_samples=int(cfg.get("jacobian_probe_n_samples", 6)),
    )
    probing_cb = ProbingCallback(
        output_path=str(run_dir / "probing.jsonl"),
        every_n_steps=probe_every,
        n_train=int(cfg.get("probing_n_train", 256)),
        n_test=int(cfg.get("probing_n_test", 64)),
    )
    latent_cov_cb = LatentCovCallback(
        output_path=str(run_dir / "latent_cov.jsonl"),
        every_n_steps=probe_every,
        n_samples=int(cfg.get("latent_cov_n_samples", 256)),
    )
    eng_cb = EngineeringMetricsCallback(
        output_path=str(run_dir / "engineering_metrics.json"),
    )
    grad_cb = PreClipGradNormCallback()
    callbacks = [object_dump_callback, jacobian_cb, probing_cb, latent_cov_cb,
                 eng_cb, grad_cb]
    # In frozen-encoder mode, add a callback that enforces eval mode on
    # encoder/projector/pred_proj across Lightning train()/eval() toggles.
    if cfg.get("freeze", {}).get("enabled", False):
        callbacks.append(KeepFrozenInEvalCallback(
            module_names=("encoder", "projector", "pred_proj")))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
