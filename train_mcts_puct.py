from __future__ import annotations

import argparse
import copy
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
            codec_version=2,
        )
    )
    net = PolicyValueNet(input_dim=codec.input_dim, hidden_dim=train_cfg.hidden_dim)
    net.to(train_cfg.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg.lr)

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
                    policy_value_net=net,
                    codec=codec,
                    sims=train_cfg.sims_per_move,
                    c_puct=train_cfg.c_puct,
                    rng=rng,
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

        # Convert final outcomes to value targets per player.
        # Lower game score is better, so training value target is normalized negative score.
        final_values = [max(-1.0, min(1.0, -float(s) / 50.0)) for s in state.scores]

        # One optimization step per episode (simple and stable baseline).
        net.train()
        optimizer.zero_grad()

        loss_policy = torch.tensor(0.0, dtype=torch.float32, device=train_cfg.device)
        loss_value = torch.tensor(0.0, dtype=torch.float32, device=train_cfg.device)

        for x, target_pi, p_idx in episode_samples:
            x = x.to(train_cfg.device).unsqueeze(0)
            target_pi = target_pi.to(train_cfg.device).unsqueeze(0)
            policy_logits, value = net(x)

            log_probs = torch.log_softmax(policy_logits, dim=-1)
            # Cross-entropy with soft targets from MCTS visit fractions.
            loss_policy = loss_policy - torch.sum(target_pi * log_probs)

            v_tgt = torch.tensor([final_values[p_idx]], dtype=torch.float32, device=train_cfg.device)
            loss_value = loss_value + torch.mean((value - v_tgt) ** 2)

        n_samples = max(1, len(episode_samples))
        loss = loss_policy / n_samples + loss_value / n_samples
        loss.backward()
        optimizer.step()

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:4d} | "
                f"loss={float(loss.item()):.4f} "
                f"policy={float((loss_policy / n_samples).item()):.4f} "
                f"value={float((loss_value / n_samples).item()):.4f} "
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
        debug_trace=bool(args.debug_trace),
    )

    run_training(game_cfg, cfg, args.output)


if __name__ == "__main__":
    main()
