from __future__ import annotations

import argparse
import copy
from collections import deque
import json
import os
import random
from dataclasses import dataclass
from typing import List

import torch

from src.engine import Engine
from src.training import StateCodec, PolicyValueNet
from src.training.state_codec import CodecConfig
from src.training.mcts_core import state_from_engine, history_view, run_root_puct


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
    lr: float = 1e-3
    hidden_dim: int = 256
    sims_per_move: int = 32
    c_puct: float = 1.5
    policy_temperature: float = 1.0
    seed: int = 42
    device: str = "cpu"
    save_every: int = 50
    history_k: int = 4
    replay_size: int = 50000
    min_replay: int = 1024
    batch_size: int = 256
    updates_per_episode: int = 8
    value_weight: float = 1.0
    entropy_weight: float = 1e-3
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    target_sync_interval: int = 20
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_eps: float = 0.25
    debug_trace: bool = False


def _load_game_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


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

    replay = deque(maxlen=max(1024, int(train_cfg.replay_size)))

    engine_cfg = copy.deepcopy(game_cfg.get("engine", {}))
    n_players = int(engine_cfg.get("n_players", 4))

    players = [ScriptedPlayer(player_idx=i) for i in range(n_players)]

    for episode in range(1, train_cfg.n_episodes + 1):
        # New game deal from engine (we reuse dealing and game constants).
        engine = Engine(engine_cfg, players)
        episode_samples: List[tuple[torch.Tensor, torch.Tensor, int]] = []

        while engine.round < engine.n_rounds:
            state = state_from_engine(engine)
            round_hist = history_view(state)
            joint = [0] * state.n_players

            for p_idx in range(state.n_players):
                hand = state.hands[p_idx]
                if len(hand) == 1:
                    joint[p_idx] = hand[0]
                    # still record a deterministic target
                    x = codec.encode_partial(hand, round_hist, p_idx)
                    target = torch.zeros(codec.cfg.n_cards, dtype=torch.float32)
                    target[hand[0] - 1] = 1.0
                    episode_samples.append((x, target, p_idx))
                    continue

                visit_dist = run_root_puct(
                    state=state,
                    player_idx=p_idx,
                    policy_value_net=search_net,
                    codec=codec,
                    sims=train_cfg.sims_per_move,
                    c_puct=train_cfg.c_puct,
                    rng=rng,
                    root_dirichlet_alpha=train_cfg.root_dirichlet_alpha,
                    root_dirichlet_eps=train_cfg.root_dirichlet_eps,
                )

                x = codec.encode_partial(hand, round_hist, p_idx)
                target = torch.zeros(codec.cfg.n_cards, dtype=torch.float32)
                for card, prob in visit_dist.items():
                    target[card - 1] = float(prob)
                episode_samples.append((x, target, p_idx))

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

        state = state_from_engine(engine)

        final_values = _blended_value_targets([float(s) for s in state.scores])
        for x, target_pi, p_idx in episode_samples:
            replay.append((x.cpu(), target_pi.cpu(), float(final_values[p_idx])))

        loss = torch.tensor(0.0)
        loss_policy = torch.tensor(0.0)
        loss_value = torch.tensor(0.0)
        loss_entropy = torch.tensor(0.0)

        if len(replay) >= max(1, train_cfg.min_replay):
            net.train()
            acc_policy = 0.0
            acc_value = 0.0
            acc_entropy = 0.0
            n_updates = max(1, int(train_cfg.updates_per_episode))

            for _ in range(n_updates):
                batch = rng.sample(list(replay), k=min(int(train_cfg.batch_size), len(replay)))
                xb = torch.stack([b[0] for b in batch], dim=0).to(train_cfg.device)
                pib = torch.stack([b[1] for b in batch], dim=0).to(train_cfg.device)
                vb = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=train_cfg.device)

                policy_logits, value = net(xb)
                log_probs = torch.log_softmax(policy_logits, dim=-1)
                probs = torch.softmax(policy_logits, dim=-1)

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
                f"replay={len(replay)} "
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
                },
            }
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(payload, output_path)
            print(f"Checkpoint saved to {output_path} (episode {episode})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline MCTS+PUCT self-play training")
    parser.add_argument("--config", required=True, help="Path to game config JSON")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--history-k", type=int, default=4)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--min-replay", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-sync-interval", type=int, default=20)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/mcts_puct_policy.pt",
        help="Output checkpoint path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_cfg = _load_game_config(args.config)

    cfg = TrainConfig(
        n_episodes=args.episodes,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        sims_per_move=args.sims,
        c_puct=args.c_puct,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        history_k=args.history_k,
        replay_size=args.replay_size,
        min_replay=args.min_replay,
        batch_size=args.batch_size,
        updates_per_episode=args.updates_per_episode,
        value_weight=args.value_weight,
        entropy_weight=args.entropy_weight,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        target_sync_interval=args.target_sync_interval,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_dirichlet_eps=args.root_dirichlet_eps,
        debug_trace=bool(args.debug_trace),
    )

    run_training(game_cfg, cfg, args.output)


if __name__ == "__main__":
    main()
