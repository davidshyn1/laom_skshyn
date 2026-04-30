import argparse
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
from tqdm import trange

from src.nn import ActionDecoder, Actor
from src.utils import DCSInMemoryDataset, create_env_from_df, normalize_img


def build_device() -> str:
    gpu_id = int(os.environ.get("GPU_ID", "0"))
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU_ID={gpu_id}. Available GPU count: {torch.cuda.device_count()}")
        return f"cuda:{gpu_id}"
    return "cpu"


def _extract_rgb_frame(obs: np.ndarray) -> np.ndarray:
    """Extract a single RGB frame from observation."""
    if obs.ndim == 3 and obs.shape[-1] >= 3:
        # For stacked observations shaped as HxWx(3*k), use latest frame.
        if obs.shape[-1] % 3 == 0:
            return obs[..., -3:].astype(np.uint8)
        return obs[..., :3].astype(np.uint8)
    raise ValueError(f"Unexpected observation shape for rendering: {obs.shape}")


@torch.no_grad()
def evaluate_bc(
    env,
    actor,
    num_episodes,
    seed=0,
    device="cpu",
    action_decoder=None,
    render_capture_dir=None,
    render_capture_count=0,
):
    returns = []
    captured = 0
    if render_capture_dir is not None and render_capture_count > 0:
        Path(render_capture_dir).mkdir(parents=True, exist_ok=True)

    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        total_reward = 0.0
        obs, _ = env.reset(seed=seed + ep)
        if captured < render_capture_count and render_capture_dir is not None:
            frame = _extract_rgb_frame(obs)
            Image.fromarray(frame).save(Path(render_capture_dir) / f"eval_frame_{captured + 1:02d}.png")
            captured += 1
        done = False
        while not done:
            obs_ = torch.tensor(obs.copy(), device=device)[None].permute(0, 3, 1, 2)
            obs_ = normalize_img(obs_)
            action, obs_emb = actor(obs_)
            if action_decoder is not None:
                if isinstance(action_decoder, ActionDecoder):
                    action = action_decoder(obs_emb, action)
                else:
                    action = action_decoder(action)

            obs, reward, terminated, truncated, _ = env.step(action.squeeze().cpu().numpy())
            if captured < render_capture_count and render_capture_dir is not None:
                frame = _extract_rgb_frame(obs)
                Image.fromarray(frame).save(Path(render_capture_dir) / f"eval_frame_{captured + 1:02d}.png")
                captured += 1
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.array(returns)


def build_actor(checkpoint: dict, cfg: dict, device: str) -> Actor:
    if "bc" not in cfg:
        raise KeyError("Config must contain `bc` section for evaluation")
    bc = cfg["bc"]
    actor_state = checkpoint["actor"]
    actor_linear_key = "actor_mean.1.weight"
    if actor_linear_key not in actor_state:
        raise KeyError(f"Checkpoint is missing `{actor_linear_key}` in actor state dict")

    # Infer actor output/action dimension from checkpoint directly.
    num_actions = actor_state[actor_linear_key].shape[0]
    data_path = bc.get("data_path")
    frame_stack = bc.get("frame_stack", 3)
    if data_path is None:
        raise KeyError("Could not determine `data_path` from config for actor build")

    dataset = DCSInMemoryDataset(data_path, frame_stack=frame_stack, device="cpu")
    actor = Actor(
        shape=(3 * frame_stack, dataset.img_hw, dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=bc.get("encoder_scale", 1),
        encoder_channels=(16, 32, 64, 128, 256) if bc.get("encoder_deep", False) else (16, 32, 32),
        encoder_num_res_blocks=bc.get("encoder_num_res_blocks", 1),
        dropout=bc.get("dropout", 0.0),
    ).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def build_action_decoder(checkpoint: dict, cfg: dict, actor: Actor, device: str):
    if "decoder" not in cfg:
        raise KeyError("Config must contain `decoder` section for evaluation")
    if "action_decoder" not in checkpoint:
        raise KeyError("Checkpoint must contain `action_decoder` for decoder evaluation")

    decoder_cfg = cfg["decoder"]
    bc = cfg["bc"]
    data_path = decoder_cfg.get("data_path", bc.get("data_path"))
    frame_stack = bc.get("frame_stack", 3)
    if data_path is None:
        raise KeyError("Could not determine `data_path` for action decoder")

    dataset = DCSInMemoryDataset(data_path, frame_stack=frame_stack, device="cpu")

    action_decoder = ActionDecoder(
        obs_emb_dim=math.prod(actor.final_encoder_shape),
        latent_act_dim=actor.num_actions,
        true_act_dim=dataset.act_dim,
        hidden_dim=decoder_cfg.get("hidden_dim", 128),
    ).to(device)
    action_decoder.load_state_dict(checkpoint["action_decoder"])
    action_decoder.eval()
    return action_decoder


def is_idm_config(cfg: dict) -> bool:
    return "idm" in cfg and "lapo" not in cfg


def resolve_eval_settings(cfg: dict, use_decoder: bool):
    if "bc" not in cfg:
        raise KeyError("Config must contain `bc` section for evaluation")
    bc = cfg["bc"]
    decoder = cfg.get("decoder", {})

    source = decoder if use_decoder else bc
    data_path = source.get("data_path", bc.get("data_path"))
    dcs_backgrounds_path = source.get("dcs_backgrounds_path", bc.get("dcs_backgrounds_path"))
    dcs_backgrounds_split = source.get("dcs_backgrounds_split", bc.get("dcs_backgrounds_split", "train"))
    frame_stack = bc.get("frame_stack", 3)
    eval_episodes = source.get("eval_episodes", bc.get("eval_episodes", 10))
    eval_seed = source.get("eval_seed", bc.get("eval_seed", 0))

    required = {
        "data_path": data_path,
        "dcs_backgrounds_path": dcs_backgrounds_path,
        "dcs_backgrounds_split": dcs_backgrounds_split,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise KeyError(f"Missing required eval config keys: {missing}")

    return {
        "data_path": data_path,
        "dcs_backgrounds_path": dcs_backgrounds_path,
        "dcs_backgrounds_split": dcs_backgrounds_split,
        "frame_stack": frame_stack,
        "eval_episodes": eval_episodes,
        "eval_seed": eval_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to yaml config")
    parser.add_argument("--model_path", default=None, help="Override model path (optional)")
    parser.add_argument(
        "--save_render_images",
        action="store_true",
        help="Save rendered evaluation frames as PNG files",
    )
    parser.add_argument(
        "--render_save_dir",
        default="data",
        help="Directory to save rendered evaluation images",
    )
    parser.add_argument(
        "--num_render_images",
        type=int,
        default=4,
        help="Number of rendered images to save",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[Stage 1/6] Loaded config: {args.config_path}")

    model_path = args.model_path or cfg.get("model_save_path")
    if not model_path:
        raise ValueError("model_save_path is empty. Set it in yaml or pass --model_path")
    model_path = str(Path(model_path))
    print(f"[Stage 2/6] Resolved checkpoint path: {model_path}")

    device = build_device()
    checkpoint = torch.load(model_path, map_location=device)
    print(f"[Stage 3/6] Loaded checkpoint on device: {device}")

    actor = build_actor(checkpoint, cfg, device)
    idm_mode = is_idm_config(cfg)
    if idm_mode:
        print("[Stage 4/6] Evaluation mode: IDM (BC only)")
        action_decoder = None
        eval_cfg = resolve_eval_settings(cfg, use_decoder=False)
    else:
        if "decoder" not in cfg:
            raise KeyError("Non-IDM evaluation requires `decoder` config section")
        print("[Stage 4/6] Evaluation mode: BC + Decoder")
        action_decoder = build_action_decoder(checkpoint, cfg, actor, device)
        eval_cfg = resolve_eval_settings(cfg, use_decoder=True)
    eval_env = create_env_from_df(
        eval_cfg["data_path"],
        eval_cfg["dcs_backgrounds_path"],
        eval_cfg["dcs_backgrounds_split"],
        frame_stack=eval_cfg["frame_stack"],
    )
    print(
        "[Stage 5/6] Built evaluation env: "
        f"episodes={eval_cfg['eval_episodes']}, seed={eval_cfg['eval_seed']}, frame_stack={eval_cfg['frame_stack']}"
    )

    eval_repeats = 1
    print(f"[Stage 6/6] Running policy evaluation ({eval_repeats} repeats)...")
    run_means = []
    all_returns = []
    for run_idx in range(eval_repeats):
        # Shift seed per repeat to sample different trajectories.
        run_seed = eval_cfg["eval_seed"] + run_idx * eval_cfg["eval_episodes"]
        should_capture = args.save_render_images and run_idx == 0
        returns = evaluate_bc(
            eval_env,
            actor,
            num_episodes=eval_cfg["eval_episodes"],
            seed=run_seed,
            device=device,
            action_decoder=action_decoder,
            render_capture_dir=args.render_save_dir if should_capture else None,
            render_capture_count=args.num_render_images if should_capture else 0,
        )
        run_means.append(float(returns.mean()))
        all_returns.append(returns)

    run_means = np.array(run_means)
    all_returns = np.concatenate(all_returns)
    mean_return_standard_error = run_means.std(ddof=1) / np.sqrt(eval_repeats)

    print(f"Loaded checkpoint: {model_path}")
    print(f"Device: {device}")
    print(f"Episodes per repeat: {eval_cfg['eval_episodes']}")
    print(f"Evaluation repeats: {eval_repeats}")
    print(f"Return mean: {all_returns.mean():.4f}")
    print(f"Return std: {all_returns.std():.4f}")
    print(f"Mean return standard error: {mean_return_standard_error:.4f}")
    if args.save_render_images:
        print(f"Saved {args.num_render_images} rendered images to: {Path(args.render_save_dir).resolve()}")


if __name__ == "__main__":
    main()
