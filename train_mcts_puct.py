from __future__ import annotations

import argparse
import copy
from collections import deque
import json
import os
import random
from dataclasses import dataclass
from typing import List

import torch  # type: ignore[reportMissingImports]

from src.engine import Engine
from src.training import StateCodec, PolicyValueNet
from src.training.state_codec import CodecConfig
from src.training.mcts_core import (
    state_from_engine,
    history_view,
    run_determinized_root_puct,
)


class ScriptedPlayer:
    """Placeholder player for Engine initialization in offline training.

    We do not use Engine.play_round() during training, but Engine still requires a player list.
    """

    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        self.next_action: int | None = None

    def set_action(self, action: int) -> None:
        self.next_action = int(action)

    def action(self, hand, history):
        if self.next_action in hand:
            return int(self.next_action)
        return min(hand)


@dataclass
class TrainConfig:
    n_episodes: int = 200
    lr: float = 7e-4
    hidden_dim: int = 256
    sims_per_move: int = 16
    c_puct: float = 1.5
    policy_temperature: float = 1.0
    seed: int = 42
    device: str = "cpu"
    save_every: int = 50
    history_k: int = 4
    replay_size: int = 4096
    min_replay: int = 4096
    batch_size: int = 128
    updates_per_episode: int = 1
    value_weight: float = 1.0
    entropy_weight: float = 1e-3
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    target_sync_interval: int = 5
    root_dirichlet_alpha: float = 0.15
    root_dirichlet_eps: float = 0.10
    determinization_samples: int = 1
    deal_seed_mode: str = "episode"
    debug_trace: bool = False
    immediate_value_weight: float = 0.30


def _load_game_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _build_train_cfg(game_cfg: dict, args: argparse.Namespace) -> TrainConfig:
    """Build training config with precedence: CLI explicit value > config.train > defaults."""
    cfg = TrainConfig()
    train_section = game_cfg.get("train", {})
    if not isinstance(train_section, dict):
        train_section = {}

    field_types = {
        "n_episodes": int,
        "lr": float,
        "hidden_dim": int,
        "sims_per_move": int,
        "c_puct": float,
        "seed": int,
        "device": str,
        "save_every": int,
        "history_k": int,
        "replay_size": int,
        "min_replay": int,
        "batch_size": int,
        "updates_per_episode": int,
        "value_weight": float,
        "entropy_weight": float,
        "grad_clip": float,
        "weight_decay": float,
        "target_sync_interval": int,
        "root_dirichlet_alpha": float,
        "root_dirichlet_eps": float,
        "determinization_samples": int,
        "deal_seed_mode": str,
        "debug_trace": bool,
        "immediate_value_weight": float,
    }

    for key, caster in field_types.items():
        if key in train_section:
            setattr(cfg, key, caster(train_section[key]))

    cli_map = {
        "episodes": "n_episodes",
        "lr": "lr",
        "hidden_dim": "hidden_dim",
        "sims": "sims_per_move",
        "c_puct": "c_puct",
        "seed": "seed",
        "device": "device",
        "save_every": "save_every",
        "history_k": "history_k",
        "replay_size": "replay_size",
        "min_replay": "min_replay",
        "batch_size": "batch_size",
        "updates_per_episode": "updates_per_episode",
        "value_weight": "value_weight",
        "entropy_weight": "entropy_weight",
        "grad_clip": "grad_clip",
        "weight_decay": "weight_decay",
        "target_sync_interval": "target_sync_interval",
        "root_dirichlet_alpha": "root_dirichlet_alpha",
        "root_dirichlet_eps": "root_dirichlet_eps",
        "determinization_samples": "determinization_samples",
        "deal_seed_mode": "deal_seed_mode",
        "immediate_value_weight": "immediate_value_weight",
    }
    for arg_name, field_name in cli_map.items():
        val = getattr(args, arg_name)
        if val is not None:
            setattr(cfg, field_name, val)

    if bool(args.debug_trace):
        cfg.debug_trace = True

    return cfg


def _sample_from_distribution(dist: dict[int, float], rng: random.Random) -> int:
    keys = list(dist.keys())
    vals = [float(dist[k]) for k in keys]
    s = sum(vals)
    if s <= 0.0:
        return min(keys)
    vals = [v / s for v in vals]
    return rng.choices(keys, weights=vals, k=1)[0]


def _rank_values(scores: List[float]) -> List[float]:
    n = len(scores)
    ranks = [0.0] * n
    for i, s in enumerate(scores):
        better = sum(1 for x in scores if x < s)
        same = sum(1 for x in scores if x == s)
        ranks[i] = (2 * better + same + 1) / 2.0

    if n <= 1:
        return [0.0] * n

    out = [0.0] * n
    for i, r in enumerate(ranks):
        out[i] = 1.0 - 2.0 * ((r - 1.0) / float(n - 1))
    return out


def _score_values(scores: List[float]) -> List[float]:
    mean_s = sum(scores) / max(1, len(scores))
    var_s = sum((x - mean_s) ** 2 for x in scores) / max(1, len(scores))
    std_s = max(1e-6, var_s ** 0.5)
    vals = []
    for s in scores:
        # Lower score is better -> negate z-score.
        z = -(float(s) - mean_s) / std_s
        vals.append(max(-1.0, min(1.0, z / 3.0)))
    return vals


def _blended_value_targets(scores: List[float]) -> List[float]:
    r = _rank_values(scores)
    z = _score_values(scores)
    out = []
    for rv, zv in zip(r, z):
        out.append(0.7 * rv + 0.3 * zv)
    return out


def _immediate_value_target(immediate_reward: float) -> float:
    # Squash immediate reward to [-1, 1] to keep value regression stable.
    return float(max(-1.0, min(1.0, immediate_reward / 6.0)))


def run_training(game_cfg: dict, train_cfg: TrainConfig, output_path: str) -> None:
    rng = random.Random(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    codec = StateCodec(
        CodecConfig(
            n_cards=int(game_cfg.get("engine", {}).get("n_cards", 104)),
            n_players=int(game_cfg.get("engine", {}).get("n_players", 4)),
            n_rounds=int(game_cfg.get("engine", {}).get("n_rounds", 10)),
            board_rows=int(game_cfg.get("engine", {}).get("board_size_y", 4)),
            board_capacity=int(game_cfg.get("engine", {}).get("board_size_x", 5)),
            history_k=int(train_cfg.history_k),
            codec_version=3,
        )
    )
    net = PolicyValueNet(input_dim=codec.input_dim, hidden_dim=train_cfg.hidden_dim)
    net.to(train_cfg.device)
    net.train()

    search_net = PolicyValueNet(input_dim=codec.input_dim, hidden_dim=train_cfg.hidden_dim)
    search_net.load_state_dict(net.state_dict())
    search_net.to(train_cfg.device)
    search_net.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    recent_buffer = deque(maxlen=max(1024, int(train_cfg.replay_size)))

    engine_cfg = copy.deepcopy(game_cfg.get("engine", {}))
    n_players = int(engine_cfg.get("n_players", 4))
    base_engine_seed = engine_cfg.get("seed", None)

    players = [ScriptedPlayer(player_idx=i) for i in range(n_players)]

    for episode in range(1, train_cfg.n_episodes + 1):
        # New game deal from engine (we reuse dealing and game constants).
        episode_engine_cfg = copy.deepcopy(engine_cfg)
        if train_cfg.deal_seed_mode == "episode":
            if base_engine_seed is None:
                episode_engine_cfg["seed"] = rng.randrange(1, 2**31 - 1)
            else:
                episode_engine_cfg["seed"] = int(base_engine_seed) + (episode - 1)
        elif train_cfg.deal_seed_mode == "random":
            episode_engine_cfg["seed"] = None
        else:
            episode_engine_cfg["seed"] = base_engine_seed

        engine = Engine(episode_engine_cfg, players)
        episode_samples: List[dict] = []

        while engine.round < engine.n_rounds:
            state = state_from_engine(engine)
            round_hist = history_view(state)
            joint = [0] * state.n_players
            round_sample_idx = [-1] * state.n_players

            for p_idx in range(state.n_players):
                hand = state.hands[p_idx]
                if len(hand) == 1:
                    joint[p_idx] = hand[0]
                    # still record a deterministic target
                    x = codec.encode_partial(hand, round_hist, p_idx)
                    target = torch.zeros(codec.cfg.n_cards, dtype=torch.float32)
                    target[hand[0] - 1] = 1.0
                    mask = codec.legal_action_mask(hand)
                    episode_samples.append(
                        {
                            "x": x,
                            "target": target,
                            "p_idx": p_idx,
                            "mask": mask,
                            "score_before": float(state.scores[p_idx]),
                            "immediate_reward": 0.0,
                        }
                    )
                    round_sample_idx[p_idx] = len(episode_samples) - 1
                    continue

                visit_dist = run_determinized_root_puct(
                    state=state,
                    player_idx=p_idx,
                    policy_value_net=search_net,
                    codec=codec,
                    total_sims=train_cfg.sims_per_move,
                    c_puct=train_cfg.c_puct,
                    rng=rng,
                    determinization_samples=train_cfg.determinization_samples,
                    root_dirichlet_alpha=train_cfg.root_dirichlet_alpha,
                    root_dirichlet_eps=train_cfg.root_dirichlet_eps,
                )

                x = codec.encode_partial(hand, round_hist, p_idx)
                target = torch.zeros(codec.cfg.n_cards, dtype=torch.float32)
                for card, prob in visit_dist.items():
                    target[card - 1] = float(prob)
                mask = codec.legal_action_mask(hand)
                episode_samples.append(
                    {
                        "x": x,
                        "target": target,
                        "p_idx": p_idx,
                        "mask": mask,
                        "score_before": float(state.scores[p_idx]),
                        "immediate_reward": 0.0,
                    }
                )
                round_sample_idx[p_idx] = len(episode_samples) - 1

                joint[p_idx] = _sample_from_distribution(visit_dist, rng)

            if train_cfg.debug_trace and (episode <= 2):
                legal_sets = {p: state.hands[p][:] for p in range(state.n_players)}
                print(
                    f"[trace][ep={episode}][round={engine.round}] "
                    f"joint={joint} legal={legal_sets} x_dim={codec.input_dim}"
                )

            for p_idx, player in enumerate(players):
                player.set_action(joint[p_idx])
            engine.play_round()
            engine.round += 1

            for p_idx in range(n_players):
                idx = round_sample_idx[p_idx]
                if idx < 0:
                    continue
                before = float(episode_samples[idx]["score_before"])
                after = float(engine.scores[p_idx])
                # Lower score is better, so taking cards creates negative immediate reward.
                episode_samples[idx]["immediate_reward"] = -(after - before)

        state = state_from_engine(engine)

        final_values = _blended_value_targets([float(s) for s in state.scores])
        ivw = max(0.0, min(1.0, float(train_cfg.immediate_value_weight)))
        for sample in episode_samples:
            p_idx = int(sample["p_idx"])
            immediate_v = _immediate_value_target(float(sample["immediate_reward"]))
            target_v = (1.0 - ivw) * float(final_values[p_idx]) + ivw * immediate_v
            recent_buffer.append(
                (
                    sample["x"].cpu(),
                    sample["target"].cpu(),
                    float(target_v),
                    sample["mask"].cpu(),
                )
            )

        loss = torch.tensor(0.0)
        loss_policy = torch.tensor(0.0)
        loss_value = torch.tensor(0.0)
        loss_entropy = torch.tensor(0.0)

        if len(recent_buffer) >= max(1, train_cfg.min_replay):
            net.train()
            acc_policy = 0.0
            acc_value = 0.0
            acc_entropy = 0.0
            n_updates = max(1, int(train_cfg.updates_per_episode))
            buffer_snapshot = list(recent_buffer)

            for _ in range(n_updates):
                batch = rng.sample(buffer_snapshot, k=min(int(train_cfg.batch_size), len(buffer_snapshot)))
                xb = torch.stack([b[0] for b in batch], dim=0).to(train_cfg.device)
                pib = torch.stack([b[1] for b in batch], dim=0).to(train_cfg.device)
                vb = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=train_cfg.device)
                legal_mask = torch.stack([b[3] for b in batch], dim=0).to(train_cfg.device)

                policy_logits, value = net(xb)
                masked_logits = policy_logits.masked_fill(legal_mask <= 0.0, -1e9)
                log_probs = torch.log_softmax(masked_logits, dim=-1)
                probs = torch.softmax(masked_logits, dim=-1)

                p_loss = -torch.mean(torch.sum(pib * log_probs, dim=-1))
                v_loss = torch.mean((value - vb) ** 2)
                entropy = -torch.mean(torch.sum(probs * log_probs, dim=-1))

                total_loss = p_loss + train_cfg.value_weight * v_loss - train_cfg.entropy_weight * entropy

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=train_cfg.grad_clip)
                optimizer.step()

                acc_policy += float(p_loss.item())
                acc_value += float(v_loss.item())
                acc_entropy += float(entropy.item())

            loss_policy = torch.tensor(acc_policy / n_updates)
            loss_value = torch.tensor(acc_value / n_updates)
            loss_entropy = torch.tensor(acc_entropy / n_updates)
            loss = loss_policy + train_cfg.value_weight * loss_value - train_cfg.entropy_weight * loss_entropy

        if episode % max(1, train_cfg.target_sync_interval) == 0:
            search_net.load_state_dict(net.state_dict())
            search_net.eval()

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:4d} | "
                f"loss={float(loss.item()):.4f} "
                f"policy={float(loss_policy.item()):.4f} "
                f"value={float(loss_value.item()):.4f} "
                f"entropy={float(loss_entropy.item()):.4f} "
                f"buffer={len(recent_buffer)} "
                f"final_scores={state.scores}"
            )

        if episode % train_cfg.save_every == 0 or episode == train_cfg.n_episodes:
            payload = {
                "model_state": net.state_dict(),
                "input_dim": codec.input_dim,
                "hidden_dim": train_cfg.hidden_dim,
                "n_cards": codec.cfg.n_cards,
                "codec": {
                    "version": codec.cfg.codec_version,
                    "history_k": codec.cfg.history_k,
                    "n_players": codec.cfg.n_players,
                    "n_rounds": codec.cfg.n_rounds,
                    "board_rows": codec.cfg.board_rows,
                    "board_capacity": codec.cfg.board_capacity,
                },
                "meta": {
                    "episode": episode,
                    "sims_per_move": train_cfg.sims_per_move,
                    "c_puct": train_cfg.c_puct,
                    "replay_size": train_cfg.replay_size,
                    "batch_size": train_cfg.batch_size,
                    "updates_per_episode": train_cfg.updates_per_episode,
                    "determinization_samples": train_cfg.determinization_samples,
                    "deal_seed_mode": train_cfg.deal_seed_mode,
                    "immediate_value_weight": train_cfg.immediate_value_weight,
                },
            }
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(payload, output_path)
            print(f"Checkpoint saved to {output_path} (episode {episode})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline MCTS+PUCT self-play training")
    parser.add_argument("--config", required=True, help="Path to game config JSON")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--history-k", type=int, default=None)
    parser.add_argument("--replay-size", type=int, default=None)
    parser.add_argument("--min-replay", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-episode", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--target-sync-interval", type=int, default=None)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=None)
    parser.add_argument("--root-dirichlet-eps", type=float, default=None)
    parser.add_argument("--determinization-samples", type=int, default=None)
    parser.add_argument(
        "--deal-seed-mode",
        type=str,
        default=None,
        choices=["episode", "fixed", "random"],
    )
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument("--immediate-value-weight", type=float, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output checkpoint path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_cfg = _load_game_config(args.config)
    cfg = _build_train_cfg(game_cfg, args)

    config_output = game_cfg.get("train", {}).get("output") if isinstance(game_cfg.get("train", {}), dict) else None
    output_path = args.output or config_output or "artifacts/mcts_puct_policy.pt"

    run_training(game_cfg, cfg, output_path)


if __name__ == "__main__":
    main()
