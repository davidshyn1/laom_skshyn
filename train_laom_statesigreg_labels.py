import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange

from src.augmentations import Augmenter
from src.nn import LAOMWithLabels
from src.scheduler import linear_annealing_with_warmup
from src.utils import DCSLAOMInMemoryDataset, DCSLAOMTrueActionsDataset, get_grad_norm, get_optim_groups, normalize_img, set_seed
from train_laom_labels import BCConfig, DecoderConfig, evaluate, train_act_decoder, train_bc

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_GPU_ID = int(os.environ.get("GPU_ID", "0"))
if torch.cuda.is_available():
    if _GPU_ID < 0 or _GPU_ID >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU_ID={_GPU_ID}. Available GPU count: {torch.cuda.device_count()}")
    DEVICE = f"cuda:{_GPU_ID}"
else:
    DEVICE = "cpu"
DEVICE_TYPE = "cuda" if DEVICE.startswith("cuda") else "cpu"

_DEFAULT_WANDB_DIR = str(Path(__file__).resolve().parent / "wandb")
_DEFAULT_TRAIN_DATA_PATH = "/yj_hdd/skshyn/lam/dataset/data/walker-run-500x-train_merged.hdf5"
_DEFAULT_LABELED_DATA_PATH = "/yj_hdd/skshyn/lam/dataset/data/walker-run-labeled-1000xtraj125.hdf5"
_DEFAULT_EVAL_DATA_PATH = "/yj_hdd/skshyn/lam/dataset/data/walker-run-10x-test.hdf5"


class SIGReg(torch.nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        proj = proj.float()
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=torch.float32)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


@dataclass
class LAOMConfig:
    num_epochs: int = 100
    batch_size: int = 256
    labeled_batch_size: int = 256
    labeled_loss_coef: float = 0.05
    sigreg_coef: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    cosine_loss: bool = False
    use_aug: bool = False
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    latent_action_dim: int = 256
    act_head_dim: int = 512
    act_head_dropout: float = 0.0
    obs_head_dim: int = 512
    obs_head_dropout: float = 0.0
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_dropout: float = 0.0
    encoder_norm_out: bool = True
    encoder_deep: bool = True
    frame_stack: int = 3
    data_path: str = _DEFAULT_TRAIN_DATA_PATH
    eval_data_path: Optional[str] = _DEFAULT_EVAL_DATA_PATH
    labeled_data_path: str = _DEFAULT_LABELED_DATA_PATH


@dataclass
class Config:
    project: str = "laom"
    group: str = "laom-statesigreg-labels"
    environment: Optional[str] = None
    name: str = "laom-statesigreg-labels"
    seed: int = 0
    wandb_dir: str = _DEFAULT_WANDB_DIR
    model_save_path: Optional[str] = None

    lapo: LAOMConfig = field(default_factory=LAOMConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"
        self.decoder.data_path = self.lapo.labeled_data_path


def train_laom(config: LAOMConfig):
    dataset = DCSLAOMInMemoryDataset(
        config.data_path, max_offset=config.future_obs_offset, frame_stack=config.frame_stack, device=DEVICE
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    labeled_dataset = DCSLAOMTrueActionsDataset(
        config.labeled_data_path,
        max_offset=config.future_obs_offset,
        frame_stack=config.frame_stack,
        device=DEVICE,
    )
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=config.labeled_batch_size, num_workers=0, pin_memory=False)

    if config.eval_data_path is not None:
        eval_dataset = DCSLAOMInMemoryDataset(config.eval_data_path, max_offset=1, frame_stack=config.frame_stack, device=DEVICE)
        eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0)

    lapo = LAOMWithLabels(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        true_act_dim=dataset.act_dim,
        latent_act_dim=config.latent_action_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
        encoder_norm_out=config.encoder_norm_out,
    ).to(DEVICE)
    torchinfo.summary(lapo, input_size=[(1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw)] * 2)
    optim = torch.optim.Adam(get_optim_groups(lapo, config.weight_decay), lr=config.learning_rate, fused=True)
    scheduler = linear_annealing_with_warmup(
        optim,
        len(dataloader) * config.warmup_epochs,
        len(dataloader) * config.num_epochs,
    )
    augmenter = Augmenter(dataset.img_hw)
    sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj).to(DEVICE)
    state_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)
    act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)
    state_act_linear_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.act_dim).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)

    total_tokens, total_iterations = 0, 0
    start_time = time.time()
    labeled_iter = iter(labeled_dataloader)
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_iterations += 1
            obs, _, future_obs, debug_actions, debug_states, _ = [b.to(DEVICE) for b in batch]
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            future_obs = normalize_img(future_obs.permute((0, 3, 1, 2)))
            if config.use_aug:
                obs = augmenter(obs)
                future_obs = augmenter(future_obs)

            with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
                latent_next_obs, latent_action, obs_hidden = lapo(obs, future_obs)
                obs_emb = lapo.encoder(obs).flatten(1)
                future_obs_emb = lapo.encoder(future_obs).flatten(1)
                if config.cosine_loss:
                    loss0 = 1 - F.cosine_similarity(latent_next_obs, future_obs_emb, dim=-1).mean()
                else:
                    loss0 = F.mse_loss(latent_next_obs, future_obs_emb)
            sigreg_loss = sigreg(obs_emb)

            labeled_batch = next(labeled_iter)
            label_obs, _, label_future_obs, label_actions, _, _ = [b.to(DEVICE) for b in labeled_batch]
            label_obs = normalize_img(label_obs.permute((0, 3, 1, 2)))
            label_future_obs = normalize_img(label_future_obs.permute((0, 3, 1, 2)))
            if config.use_aug:
                label_obs = augmenter(label_obs)
                label_future_obs = augmenter(label_future_obs)
            with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
                _, _, pred_action, _ = lapo(label_obs, label_future_obs, predict_true_act=True)
                loss1 = F.mse_loss(pred_action, label_actions)

            loss = loss0 + config.labeled_loss_coef * loss1 + config.sigreg_coef * sigreg_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()

            with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, debug_states)
            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
                pred_action = act_linear_probe(latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action, debug_actions)
            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()

            with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
                state_pred_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(state_pred_action, debug_actions)
            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()

            wandb.log(
                {
                    "lapo/total_loss": loss.item(),
                    "lapo/mse_loss": loss0.item(),
                    "lapo/true_action_mse_loss": loss1.item(),
                    "lapo/sigreg_loss": sigreg_loss.item(),
                    "lapo/state_probe_mse_loss": state_probe_loss.item(),
                    "lapo/action_probe_mse_loss": act_probe_loss.item(),
                    "lapo/state_action_probe_mse_loss": state_act_probe_loss.item(),
                    "lapo/obs_hidden_norm": torch.norm(obs_hidden, p=2, dim=-1).mean().item(),
                    "lapo/future_obs_norm": torch.norm(future_obs_emb, p=2, dim=-1).mean().item(),
                    "lapo/online_obs_norm": torch.norm(latent_next_obs, p=2, dim=-1).mean().item(),
                    "lapo/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                    "lapo/throughput": total_tokens / (time.time() - start_time),
                    "lapo/learning_rate": scheduler.get_last_lr()[0],
                    "lapo/grad_norm": get_grad_norm(lapo).item(),
                    "lapo/epoch": epoch,
                    "lapo/total_steps": total_iterations,
                }
            )
        if config.eval_data_path is not None:
            eval_mse_loss = evaluate(lapo, eval_dataloader, device=DEVICE)
            wandb.log(
                {
                    "lapo/total_steps": total_iterations,
                    "lapo/eval_true_action_mse_loss": eval_mse_loss,
                    "lapo/epoch": epoch,
                    "lapo/total_steps": total_iterations,
                }
            )
    return lapo


@pyrallis.wrap()
def train(config: Config):
    wandb_project = f"{config.project}-{config.environment}" if config.environment else config.project
    run = wandb.init(
        project=wandb_project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
        dir=config.wandb_dir,
    )
    set_seed(config.seed)
    lapo = train_laom(config=config.lapo)
    wandb.log({"bc/total_steps": 0})
    actor = train_bc(lam=lapo, config=config.bc)
    wandb.log({"decoder/total_steps": 0})
    action_decoder = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc)

    if config.model_save_path:
        save_path = Path(config.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"lapo": lapo.state_dict(), "actor": actor.state_dict(), "action_decoder": action_decoder.state_dict()},
            save_path,
        )
        print(f"Saved model checkpoint to: {save_path}")

    run.finish()
    return lapo, actor, action_decoder


if __name__ == "__main__":
    train()
