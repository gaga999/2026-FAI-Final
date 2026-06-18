import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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

FAST_TEACHER_ARGS = {
    # ("src.players.b13902102.best_player1", "BestPlayer1"): {
    #     "deadline_s": 0.2,
    #     "rollout_min": 8,
    #     "rollout_max": 24,
    #     "cand_keep": 6,
    # },
    # ("src.players.b13902102.best_player2", "BestPlayer2"): {
    #     "deadline_s": 0.2,
    #     "rollout_min": 8,
    #     "rollout_max": 24,
    #     "cand_keep": 6,
    # },
    ("src.players.b13902102.adaptive_portfolio_player", "AdaptivePortfolioPlayer"): {
        "deadline_s": 0.2,
        # "samples_min": 8,
        # "samples_max": 24,
        # "cand_keep": 6,
    },
    ("src.players.b13902102.rollout_search_player", "RolloutSearchPlayer"): {
        "deadline_s": 0.2,
    },
    ("src.players.b13902102.beam_expectimax_player", "BeamExpectimaxPlayer"): {
        "deadline_s": 0.2,
    },
}


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


def instantiate_teacher(path, cls_name, player_idx, fast_teachers=True):
    module = importlib.import_module(path)
    cls = getattr(module, cls_name)
    args = FAST_TEACHER_ARGS.get((path, cls_name), {}) if fast_teachers else {}
    return cls(player_idx=player_idx, **args)


def sample_teacher_specs(rng, n_players=4):
    weights = [spec[2] for spec in DEFAULT_TEACHERS]
    selected = rng.choices(DEFAULT_TEACHERS, weights=weights, k=n_players)
    return selected


def _records_to_tensors(records):
    return {
        "features": torch.tensor([row.features for row in records], dtype=torch.float32),
        "legal_masks": torch.tensor([row.legal_mask for row in records], dtype=torch.bool),
        "actions": torch.tensor([row.action_idx for row in records], dtype=torch.long),
    }


def _save_dataset_checkpoint(output_path, existing_data, records, metadata):
    new_data = _records_to_tensors(records)
    if existing_data is None:
        payload = {**new_data, "metadata": metadata}
    elif records:
        payload = {
            "features": torch.cat((existing_data["features"], new_data["features"]), dim=0),
            "legal_masks": torch.cat((existing_data["legal_masks"], new_data["legal_masks"]), dim=0),
            "actions": torch.cat((existing_data["actions"], new_data["actions"]), dim=0),
            "metadata": metadata,
        }
    else:
        payload = {**existing_data, "metadata": metadata}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    return payload


def collect_bc_dataset(
    num_games,
    output_path,
    seed=0,
    fast_teachers=True,
    existing_dataset_path=None,
    save_every_games=100,
):
    rng = random.Random(seed)
    records = []
    output_path = Path(output_path)
    existing_data = None
    existing_games = 0
    existing_metadata = {}
    if existing_dataset_path is not None:
        existing_dataset_path = Path(existing_dataset_path)
        print(f"[BC] Loading existing dataset: {existing_dataset_path}", flush=True)
        existing_data = torch.load(existing_dataset_path, map_location="cpu")
        existing_metadata = dict(existing_data.get("metadata", {}))
        existing_games = int(existing_metadata.get("num_games", 0))
        print(f"[BC] Existing records: {len(existing_data['actions'])}", flush=True)

    save_every_games = max(1, int(save_every_games))
    teacher_mode = "fast" if fast_teachers else "full"
    print(
        f"[BC] Collecting teacher decisions: games={int(num_games)}, seed={seed}, "
        f"teacher_mode={teacher_mode}, save_every_games={save_every_games}",
        flush=True,
    )
    progress = tqdm(range(int(num_games)), desc="BC collect games", unit="game")
    for game_idx in progress:
        teacher_specs = sample_teacher_specs(rng)
        players = []
        for seat, (path, cls_name, _) in enumerate(teacher_specs):
            teacher = instantiate_teacher(path, cls_name, seat, fast_teachers=fast_teachers)
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
        progress.set_postfix(records=len(records), teachers=",".join(spec[1] for spec in teacher_specs))

        completed_games = game_idx + 1
        if completed_games % save_every_games == 0:
            metadata = {
                **existing_metadata,
                "num_games": existing_games + completed_games,
                "seed": seed,
                "teacher_pool": DEFAULT_TEACHERS,
                "teacher_mode": teacher_mode,
                "last_collection_games": completed_games,
            }
            existing_data = _save_dataset_checkpoint(output_path, existing_data, records, metadata)
            records.clear()
            tqdm.write(
                f"[BC] Dataset checkpoint: games={completed_games}/{int(num_games)}, "
                f"records={len(existing_data['actions'])} -> {output_path}"
            )

    completed_games = int(num_games)
    metadata = {
        **existing_metadata,
        "num_games": existing_games + completed_games,
        "seed": seed,
        "teacher_pool": DEFAULT_TEACHERS,
        "teacher_mode": teacher_mode,
        "last_collection_games": completed_games,
    }
    if records or existing_data is None or completed_games % save_every_games != 0:
        existing_data = _save_dataset_checkpoint(output_path, existing_data, records, metadata)
    print(f"[BC] Dataset ready: records={len(existing_data['actions'])} -> {output_path}", flush=True)
    return output_path


def train_bc(dataset_path, checkpoint_path, epochs=5, batch_size=256, lr=5.0e-4, entropy_coef=0.003, label_smoothing=0.02):
    print(f"[BC] Loading dataset: {dataset_path}", flush=True)
    data = torch.load(Path(dataset_path), map_location="cpu")
    dataset = TensorDataset(data["features"], data["legal_masks"], data["actions"])
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    print(
        f"[BC] Training policy: samples={len(dataset)}, epochs={int(epochs)}, batch_size={int(batch_size)}, lr={float(lr)}",
        flush=True,
    )
    model = PolicyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    epoch_bar = tqdm(range(1, int(epochs) + 1), desc="BC epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        loss_sum = 0.0
        batch_count = 0
        batch_bar = tqdm(loader, desc=f"BC epoch {epoch}/{int(epochs)}", unit="batch", leave=False)
        for features, legal_masks, actions in batch_bar:
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
            loss_value = float(loss.detach().cpu())
            loss_sum += loss_value
            batch_count += 1
            batch_bar.set_postfix(loss=f"{loss_value:.4f}")
        avg_loss = loss_sum / max(1, batch_count)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        tqdm.write(f"[BC] Epoch {epoch}/{int(epochs)} done: avg_loss={avg_loss:.4f}")
        if epoch % 100 == 0:
            save_policy_checkpoint(checkpoint_path, model, optimizer, stage="bc", dataset=str(dataset_path))
    print(f"[BC] Saving checkpoint: {checkpoint_path}", flush=True)
    save_policy_checkpoint(checkpoint_path, model, optimizer, stage="bc", dataset=str(dataset_path))


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning warmup for b13902102 policy model.")
    parser.add_argument("--num-games", type=int, default=2000)
    parser.add_argument("--dataset", type=str, default="artifacts/b13902102/bc_dataset.pt")
    parser.add_argument(
        "--existing-dataset",
        type=str,
        default=None,
        help="Load this dataset first and append newly collected decisions to --dataset.",
    )
    parser.add_argument(
        "--save-every-games",
        type=int,
        default=1000,
        help="Checkpoint the collected dataset after this many completed games.",
    )
    parser.add_argument("--checkpoint", type=str, default="artifacts/b13902102/bc_policy.pth")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-teachers", action="store_true", help="Use full default teacher budgets instead of fast BC collection budgets.")
    args = parser.parse_args()
    print("[BC] Stage 1/2: dataset collection", flush=True)
    collect_bc_dataset(
        args.num_games,
        args.dataset,
        seed=args.seed,
        fast_teachers=not args.full_teachers,
        existing_dataset_path=args.existing_dataset,
        save_every_games=args.save_every_games,
    )
    print("[BC] Stage 2/2: supervised training", flush=True)
    train_bc(args.dataset, args.checkpoint, epochs=args.epochs)
    print("[BC] Done", flush=True)


if __name__ == "__main__":
    main()
