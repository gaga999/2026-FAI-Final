import argparse
import importlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from src.engine import Engine
from src.players.b13902102.rl_features import action_index, extract_policy_features
from src.players.b13902102.rl_model import PolicyMLP, mask_logits, save_policy_checkpoint


RANK_WEIGHTS = (1.70, 1.15, 0.75, 0.40)

# module, class, collection sampling weight, optional loss quality weight
WEIGHTED_TEACHERS = [
    ("src.players.b13902102.best_player1", "BestPlayer1", 4.0, 3.25),
    ("src.players.b13902102.best_player2", "BestPlayer2", 2.0, 2.75),
    ("src.players.b13902102.adaptive_portfolio_player", "AdaptivePortfolioPlayer", 2.0, 1.15),
    ("src.players.b13902102.rollout_search_player", "RolloutSearchPlayer", 4.0, 4.10),
    ("src.players.b13902102.beam_expectimax_player", "BeamExpectimaxPlayer", 1.0, 0.95),
    ("src.players.TA.public_baselines2", "Baseline6", 5.0, 5.00),
    ("src.players.TA.public_baselines2", "Baseline7", 5.0, 5.00),
    ("src.players.TA.public_baselines2", "Baseline8", 3.0, 4.00),
    ("src.players.TA.public_baselines2", "Baseline9", 5.0, 5.00),
    ("src.players.TA.public_baselines2", "Baseline10", 6.0, 10.00),
    ("src.players.b13902102.risk_heuristic_player", "RiskHeuristicPlayer", 0.2, 0.65),
    ("src.players.TA.public_baselines1", "Baseline1", 0.35, 0.50),
    ("src.players.TA.public_baselines1", "Baseline2", 0.35, 0.50),
    ("src.players.TA.public_baselines1", "Baseline3", 0.35, 0.50),
    ("src.players.TA.public_baselines1", "Baseline4", 2.5, 1.00),
    ("src.players.TA.public_baselines1", "Baseline5", 2.5, 1.00),
]

FAST_TEACHER_ARGS = {
    # ("src.players.b13902102.best_player1", "BestPlayer1"): {
    #     "deadline_s": 0.25,
    #     "rollout_min": 8,
    #     "rollout_max": 24,
    #     "cand_keep": 6,
    # },
    # ("src.players.b13902102.best_player2", "BestPlayer2"): {
    #     "deadline_s": 0.25,
    #     "rollout_min": 8,
    #     "rollout_max": 24,
    #     "cand_keep": 6,
    # },
    ("src.players.b13902102.adaptive_portfolio_player", "AdaptivePortfolioPlayer"): {
        "deadline_s": 0.20,
        # "samples_min": 16,
        # "samples_max": 32,
        # "cand_keep": 6,
    },
    ("src.players.b13902102.rollout_search_player", "RolloutSearchPlayer"): {
        "deadline_s": 0.20,
    },
    ("src.players.b13902102.beam_expectimax_player", "BeamExpectimaxPlayer"): {
        "deadline_s": 0.20,
    },
}

TEACHER_IDS = {
    (path, cls_name): index
    for index, (path, cls_name, _, _) in enumerate(WEIGHTED_TEACHERS)
}

WEIGHTED_BC_CONFIG = {
    "num_games": 2000,
    "save_every_games": 1000,
    "epochs": 5000,
    "batch_size": 256,
    "learning_rate": 5.0e-4,
    "entropy_coef": 0.003,
    "label_smoothing": 0.02,
    "score_weight_coef": 0.025,
    "teacher_weight_strength": 0.5,
}


@dataclass
class RecordedDecision:
    features: object
    legal_mask: object
    action_idx: int


class RecordingPlayer:
    def __init__(self, player, player_idx, records):
        self.player = player
        self.player_idx = int(player_idx)
        self.records = records

    def action(self, hand, history):
        features, legal_mask = extract_policy_features(
            hand,
            history,
            self.player_idx,
        )
        card = self.player.action(hand, history)
        if isinstance(card, int) and card in hand:
            self.records.append(
                RecordedDecision(
                    features,
                    legal_mask,
                    action_index(card),
                )
            )
        return card


def instantiate_teacher(path, cls_name, player_idx, fast_teachers=True):
    module = importlib.import_module(path)
    teacher_class = getattr(module, cls_name)
    kwargs = FAST_TEACHER_ARGS.get((path, cls_name), {}) if fast_teachers else {}
    return teacher_class(player_idx=player_idx, **kwargs)


def sample_teacher_specs(rng, n_players=4):
    sampling_weights = [spec[2] for spec in WEIGHTED_TEACHERS]
    return rng.choices(
        WEIGHTED_TEACHERS,
        weights=sampling_weights,
        k=n_players,
    )


@dataclass
class WeightedDecision:
    features: object
    legal_mask: object
    action_idx: int
    weight: float
    outcome_weight: float
    teacher_weight: float
    rank: float
    final_score: float
    teacher_id: int


def fractional_ranks(scores):
    ranks = []
    for score in scores:
        better_count = sum(1 for other in scores if other < score)
        same_count = sum(1 for other in scores if other == score)
        ranks.append((2 * better_count + same_count + 1) / 2.0)
    return ranks


def rank_weight(rank):
    position = float(rank) - 1.0
    if not 0.0 <= position <= len(RANK_WEIGHTS) - 1:
        raise ValueError(f"Rank must be between 1 and {len(RANK_WEIGHTS)}")
    lower = int(position)
    upper = min(lower + 1, len(RANK_WEIGHTS) - 1)
    fraction = position - lower
    return (1.0 - fraction) * RANK_WEIGHTS[lower] + fraction * RANK_WEIGHTS[upper]


def teacher_quality_weight(teacher_spec, strength=0.5):
    path, cls_name, _, base = teacher_spec
    weight = 1.0 + float(strength) * (base - 1.0)
    if weight <= 0.0:
        raise ValueError(
            f"Teacher weight must stay positive for {(path, cls_name)}; "
            f"got {weight} from strength={strength}"
        )
    return weight


def game_player_weights(
    scores,
    teacher_specs,
    score_weight_coef=0.025,
    teacher_weight_strength=0.5,
):
    ranks = fractional_ranks(scores)
    mean_score = sum(scores) / max(1, len(scores))
    raw_weights = []
    components = []

    for score, rank, teacher_spec in zip(scores, ranks, teacher_specs):
        outcome = rank_weight(rank)
        score_factor = math.exp(-float(score_weight_coef) * (float(score) - mean_score))
        teacher = teacher_quality_weight(teacher_spec, teacher_weight_strength)
        raw_weights.append(outcome * score_factor * teacher)
        components.append((outcome * score_factor, teacher))

    total_weight = sum(raw_weights)
    if total_weight <= 0.0:
        raise ValueError("Weighted BC game produced a non-positive total weight")
    normalization = len(raw_weights) / total_weight
    final_weights = [weight * normalization for weight in raw_weights]
    return ranks, final_weights, components


def _records_to_tensors(records):
    return {
        "features": torch.tensor([row.features for row in records], dtype=torch.float32),
        "legal_masks": torch.tensor([row.legal_mask for row in records], dtype=torch.bool),
        "actions": torch.tensor([row.action_idx for row in records], dtype=torch.long),
        "weights": torch.tensor([row.weight for row in records], dtype=torch.float32),
        "outcome_weights": torch.tensor([row.outcome_weight for row in records], dtype=torch.float32),
        "teacher_weights": torch.tensor([row.teacher_weight for row in records], dtype=torch.float32),
        "ranks": torch.tensor([row.rank for row in records], dtype=torch.float32),
        "final_scores": torch.tensor([row.final_score for row in records], dtype=torch.float32),
        "teacher_ids": torch.tensor([row.teacher_id for row in records], dtype=torch.long),
    }


def _normalize_existing_dataset(data):
    size = len(data["actions"])
    normalized = dict(data)
    normalized.setdefault("weights", torch.ones(size, dtype=torch.float32))
    normalized.setdefault("outcome_weights", torch.ones(size, dtype=torch.float32))
    normalized.setdefault("teacher_weights", torch.ones(size, dtype=torch.float32))
    normalized.setdefault("ranks", torch.zeros(size, dtype=torch.float32))
    normalized.setdefault("final_scores", torch.zeros(size, dtype=torch.float32))
    normalized.setdefault("teacher_ids", torch.full((size,), -1, dtype=torch.long))
    return normalized


def _save_dataset_checkpoint(output_path, existing_data, records, metadata):
    if records:
        new_data = _records_to_tensors(records)
        if existing_data is None:
            payload = {**new_data, "metadata": metadata}
        else:
            payload = {
                key: torch.cat((existing_data[key], new_data[key]), dim=0)
                for key in new_data
            }
            payload["metadata"] = metadata
    elif existing_data is not None:
        payload = {**existing_data, "metadata": metadata}
    else:
        raise ValueError("Cannot create an empty weighted BC dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    return payload


def collect_weighted_bc_dataset(
    num_games,
    output_path,
    seed=0,
    fast_teachers=True,
    existing_dataset_path=None,
    save_every_games=100,
    score_weight_coef=0.025,
    teacher_weight_strength=0.5,
):
    rng = random.Random(seed)
    records = []
    output_path = Path(output_path)
    existing_data = None
    existing_games = 0
    existing_metadata = {}

    if existing_dataset_path is not None:
        existing_dataset_path = Path(existing_dataset_path)
        print(f"[Weighted BC] Loading existing dataset: {existing_dataset_path}", flush=True)
        existing_data = _normalize_existing_dataset(
            torch.load(existing_dataset_path, map_location="cpu")
        )
        existing_metadata = dict(existing_data.get("metadata", {}))
        existing_games = int(existing_metadata.get("num_games", 0))
        print(f"[Weighted BC] Existing records: {len(existing_data['actions'])}", flush=True)

    save_every_games = max(1, int(save_every_games))
    teacher_mode = "fast" if fast_teachers else "full"
    print(
        f"[Weighted BC] Collecting: games={int(num_games)}, seed={seed}, "
        f"teacher_mode={teacher_mode}, save_every_games={save_every_games}, "
        f"configured_fast_teachers={len(FAST_TEACHER_ARGS) if fast_teachers else 0}",
        flush=True,
    )

    progress = tqdm(range(int(num_games)), desc="Weighted BC collect", unit="game")
    for game_idx in progress:
        teacher_specs = sample_teacher_specs(rng)
        seat_records = [[] for _ in teacher_specs]
        players = []
        for seat, (path, cls_name, _, _) in enumerate(teacher_specs):
            teacher = instantiate_teacher(path, cls_name, seat, fast_teachers=fast_teachers)
            players.append(RecordingPlayer(teacher, seat, seat_records[seat]))

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
        scores, _ = engine.play_game()
        ranks, player_weights, components = game_player_weights(
            scores,
            teacher_specs,
            score_weight_coef=score_weight_coef,
            teacher_weight_strength=teacher_weight_strength,
        )

        for seat, decisions in enumerate(seat_records):
            outcome_weight, teacher_weight = components[seat]
            teacher_key = teacher_specs[seat][:2]
            for decision in decisions:
                records.append(
                    WeightedDecision(
                        features=decision.features,
                        legal_mask=decision.legal_mask,
                        action_idx=decision.action_idx,
                        weight=player_weights[seat],
                        outcome_weight=outcome_weight,
                        teacher_weight=teacher_weight,
                        rank=ranks[seat],
                        final_score=scores[seat],
                        teacher_id=TEACHER_IDS.get(teacher_key, -1),
                    )
                )

        completed_games = game_idx + 1
        progress.set_postfix(
            records=len(records),
            scores=",".join(str(score) for score in scores),
        )
        if completed_games % save_every_games == 0:
            metadata = _dataset_metadata(
                existing_metadata,
                existing_games + completed_games,
                completed_games,
                seed,
                teacher_mode,
                score_weight_coef,
                teacher_weight_strength,
            )
            existing_data = _save_dataset_checkpoint(
                output_path, existing_data, records, metadata
            )
            records.clear()
            tqdm.write(
                f"[Weighted BC] Dataset checkpoint: games={completed_games}/{int(num_games)}, "
                f"records={len(existing_data['actions'])} -> {output_path}"
            )

    completed_games = int(num_games)
    metadata = _dataset_metadata(
        existing_metadata,
        existing_games + completed_games,
        completed_games,
        seed,
        teacher_mode,
        score_weight_coef,
        teacher_weight_strength,
    )
    if records or existing_data is None or completed_games % save_every_games != 0:
        existing_data = _save_dataset_checkpoint(
            output_path, existing_data, records, metadata
        )
    print(
        f"[Weighted BC] Dataset ready: records={len(existing_data['actions'])} -> {output_path}",
        flush=True,
    )
    return output_path


def _dataset_metadata(
    existing_metadata,
    num_games,
    last_collection_games,
    seed,
    teacher_mode,
    score_weight_coef,
    teacher_weight_strength,
):
    return {
        **existing_metadata,
        "weighted": True,
        "num_games": int(num_games),
        "last_collection_games": int(last_collection_games),
        "seed": int(seed),
        "teacher_pool": WEIGHTED_TEACHERS,
        "teacher_mode": teacher_mode,
        "fast_teacher_args": (
            FAST_TEACHER_ARGS
            if teacher_mode == "fast"
            else {}
        ),
        "rank_weights": RANK_WEIGHTS,
        "score_weight_coef": float(score_weight_coef),
        "teacher_weight_strength": float(teacher_weight_strength),
        "teacher_ids": TEACHER_IDS,
    }


def weighted_supervised_ce_loss(
    logits,
    legal_mask,
    action_indices,
    sample_weights,
    label_smoothing=0.02,
    entropy_coef=0.0,
):
    masked = mask_logits(logits, legal_mask)
    log_probs = F.log_softmax(masked, dim=-1)
    legal_float = legal_mask.float()
    legal_count = legal_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    smooth_target = legal_float / legal_count
    hard_target = F.one_hot(action_indices, num_classes=masked.shape[-1]).float()
    target = (1.0 - float(label_smoothing)) * hard_target
    target = target + float(label_smoothing) * smooth_target

    ce_per_sample = -(target * log_probs).sum(dim=-1)
    probs = torch.exp(log_probs)
    entropy_per_sample = -(probs * log_probs).sum(dim=-1)
    weights = sample_weights.float()
    if bool(torch.any(weights < 0.0).item()):
        raise ValueError("Weighted BC sample weights must be non-negative")
    denominator = weights.sum().clamp_min(1.0e-8)
    ce = (weights * ce_per_sample).sum() / denominator
    entropy = (weights * entropy_per_sample).sum() / denominator
    return ce - float(entropy_coef) * entropy, {
        "ce": ce.detach(),
        "entropy": entropy.detach(),
        "mean_weight": weights.mean().detach(),
    }


def train_weighted_bc(
    dataset_path,
    checkpoint_path,
    epochs=5,
    batch_size=256,
    lr=5.0e-4,
    entropy_coef=0.003,
    label_smoothing=0.02,
):
    print(f"[Weighted BC] Loading dataset: {dataset_path}", flush=True)
    data = _normalize_existing_dataset(
        torch.load(Path(dataset_path), map_location="cpu")
    )
    dataset = TensorDataset(
        data["features"],
        data["legal_masks"],
        data["actions"],
        data["weights"],
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    print(
        f"[Weighted BC] Training: samples={len(dataset)}, epochs={int(epochs)}, "
        f"batch_size={int(batch_size)}, lr={float(lr)}",
        flush=True,
    )

    print("[Weighted BC] Initializing policy with random weights", flush=True)
    model = PolicyMLP()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=1.0e-4
    )
    checkpoint_metadata = {
        "stage": "weighted_bc",
        "dataset": str(dataset_path),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "entropy_coef": float(entropy_coef),
        "label_smoothing": float(label_smoothing),
    }
    epoch_bar = tqdm(range(1, int(epochs) + 1), desc="Weighted BC epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        loss_sum = 0.0
        batch_count = 0
        batch_bar = tqdm(
            loader,
            desc=f"Weighted BC epoch {epoch}/{int(epochs)}",
            unit="batch",
            leave=False,
        )
        for features, legal_masks, actions, weights in batch_bar:
            logits = model(features)
            loss, metrics = weighted_supervised_ce_loss(
                logits,
                legal_masks,
                actions,
                weights,
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
            batch_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                weight=f"{float(metrics['mean_weight'].cpu()):.3f}",
            )

        avg_loss = loss_sum / max(1, batch_count)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        tqdm.write(
            f"[Weighted BC] Epoch {epoch}/{int(epochs)} done: avg_loss={avg_loss:.4f}"
        )
        if epoch % 100 == 0:
            save_policy_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                **checkpoint_metadata,
            )

    print(f"[Weighted BC] Saving checkpoint: {checkpoint_path}", flush=True)
    save_policy_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        **checkpoint_metadata,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Outcome- and teacher-weighted behavior cloning."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=WEIGHTED_BC_CONFIG["num_games"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="artifacts/b13902102/bc_weighted_dataset.pt",
    )
    parser.add_argument(
        "--existing-dataset",
        type=str,
        default=None,
        help="Load this dataset first and append newly collected decisions to --dataset.",
    )
    parser.add_argument(
        "--save-every-games",
        type=int,
        default=WEIGHTED_BC_CONFIG["save_every_games"],
        help="Checkpoint the collected dataset after this many completed games.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/b13902102/bc_weighted_policy.pth",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=WEIGHTED_BC_CONFIG["epochs"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=WEIGHTED_BC_CONFIG["batch_size"],
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=WEIGHTED_BC_CONFIG["learning_rate"],
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=WEIGHTED_BC_CONFIG["entropy_coef"],
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=WEIGHTED_BC_CONFIG["label_smoothing"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score-weight-coef",
        type=float,
        default=WEIGHTED_BC_CONFIG["score_weight_coef"],
    )
    parser.add_argument(
        "--teacher-weight-strength",
        type=float,
        default=WEIGHTED_BC_CONFIG["teacher_weight_strength"],
    )
    parser.add_argument(
        "--full-teachers",
        action="store_true",
        help="Use full default teacher budgets instead of fast collection budgets.",
    )
    args = parser.parse_args()
    if args.save_every_games < 1:
        parser.error("--save-every-games must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if not 0.0 <= args.label_smoothing <= 1.0:
        parser.error("--label-smoothing must be between 0 and 1")

    print("[Weighted BC] Stage 1/2: dataset collection", flush=True)
    collect_weighted_bc_dataset(
        args.num_games,
        args.dataset,
        seed=args.seed,
        fast_teachers=not args.full_teachers,
        existing_dataset_path=args.existing_dataset,
        save_every_games=args.save_every_games,
        score_weight_coef=args.score_weight_coef,
        teacher_weight_strength=args.teacher_weight_strength,
    )
    print("[Weighted BC] Stage 2/2: supervised training", flush=True)
    train_weighted_bc(
        args.dataset,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        entropy_coef=args.entropy_coef,
        label_smoothing=args.label_smoothing,
    )
    print("[Weighted BC] Done", flush=True)


if __name__ == "__main__":
    main()
