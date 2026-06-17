import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from src.engine import Engine
from src.players.b13902102.rl_features import action_index, extract_policy_features
from src.players.b13902102.rl_model import PolicyMLP, save_policy_checkpoint, supervised_ce_loss


DEFAULT_TEACHERS = [
    ("src.players.b13902102.best_player1", "BestPlayer1", 4.0),
    ("src.players.b13902102.best_player2", "BestPlayer2", 2.0),
    ("src.players.b13902102.adaptive_portfolio_player", "AdaptivePortfolioPlayer", 2.0),
    ("src.players.b13902102.rollout_search_player", "RolloutSearchPlayer", 4.0),
    ("src.players.b13902102.beam_expectimax_player", "BeamExpectimaxPlayer", 1.0),
    ("src.players.TA.public_baselines2", "Baseline6", 5.0),
    ("src.players.TA.public_baselines2", "Baseline7", 5.0),
    ("src.players.TA.public_baselines2", "Baseline8", 3.0),
    ("src.players.TA.public_baselines2", "Baseline9", 5.0),
    ("src.players.TA.public_baselines2", "Baseline10", 6.0),
    ("src.players.b13902102.risk_heuristic_player", "RiskHeuristicPlayer", 0.2),
    ("src.players.TA.public_baselines1", "Baseline1", 0.35),
    ("src.players.TA.public_baselines1", "Baseline2", 0.35),
    ("src.players.TA.public_baselines1", "Baseline3", 0.35),
    ("src.players.TA.public_baselines1", "Baseline4", 2.5),
    ("src.players.TA.public_baselines1", "Baseline5", 2.5),
]


@dataclass
class RecordedDecision:
    features: object
    legal_mask: object
    action_idx: int


class RecordingPlayer:
    def __init__(self, player, player_idx, records):
        self.player = player
        self.player_idx = player_idx
        self.records = records

    def action(self, hand, history):
        features, legal_mask = extract_policy_features(hand, history, self.player_idx)
        card = self.player.action(hand, history)
        if isinstance(card, int) and card in hand:
            self.records.append(RecordedDecision(features, legal_mask, action_index(card)))
        return card


def instantiate_teacher(path, cls_name, player_idx):
    module = importlib.import_module(path)
    cls = getattr(module, cls_name)
    return cls(player_idx=player_idx)


def sample_teacher_specs(rng, n_players=4):
    weights = [spec[2] for spec in DEFAULT_TEACHERS]
    selected = rng.choices(DEFAULT_TEACHERS, weights=weights, k=n_players)
    return selected


def collect_bc_dataset(num_games, output_path, seed=0):
    rng = random.Random(seed)
    records = []
    for game_idx in range(int(num_games)):
        teacher_specs = sample_teacher_specs(rng)
        players = []
        for seat, (path, cls_name, _) in enumerate(teacher_specs):
            teacher = instantiate_teacher(path, cls_name, seat)
            players.append(RecordingPlayer(teacher, seat, records))
        engine = Engine(
            {
                "n_players": 4,
                "n_rounds": 10,
                "verbose": False,
                "seed": rng.randrange(10**9),
                "timeout": None,
            },
            players,
        )
        engine.play_game()

    payload = {
        "features": torch.tensor([row.features for row in records], dtype=torch.float32),
        "legal_masks": torch.tensor([row.legal_mask for row in records], dtype=torch.bool),
        "actions": torch.tensor([row.action_idx for row in records], dtype=torch.long),
        "metadata": {"num_games": int(num_games), "seed": seed, "teacher_pool": DEFAULT_TEACHERS},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def train_bc(dataset_path, checkpoint_path, epochs=5, batch_size=256, lr=5.0e-4, entropy_coef=0.003, label_smoothing=0.02):
    data = torch.load(Path(dataset_path), map_location="cpu")
    dataset = TensorDataset(data["features"], data["legal_masks"], data["actions"])
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    model = PolicyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    for _ in range(int(epochs)):
        model.train()
        for features, legal_masks, actions in loader:
            logits = model(features)
            loss, _ = supervised_ce_loss(
                logits,
                legal_masks,
                actions,
                label_smoothing=float(label_smoothing),
                entropy_coef=float(entropy_coef),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    save_policy_checkpoint(checkpoint_path, model, optimizer, stage="bc", dataset=str(dataset_path))


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning warmup for b13902102 policy model.")
    parser.add_argument("--num-games", type=int, default=2000)
    parser.add_argument("--dataset", type=str, default="artifacts/b13902102/bc_dataset.pt")
    parser.add_argument("--checkpoint", type=str, default="artifacts/b13902102/bc_policy.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    collect_bc_dataset(args.num_games, args.dataset, seed=args.seed)
    train_bc(args.dataset, args.checkpoint, epochs=args.epochs)


if __name__ == "__main__":
    main()
