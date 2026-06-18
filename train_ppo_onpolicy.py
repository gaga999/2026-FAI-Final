import argparse
import importlib
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.players.b13902102.rl_checkpoint import load_checkpoint, save_checkpoint
from src.players.b13902102.rl_features import extract_policy_features
from src.players.b13902102.rl_model import PolicyMLP, mask_logits
from src.players.b13902102.rl_train_env import deal_light_game


# Same (module, class, weight) format as train_bc.DEFAULT_TEACHERS.
BASELINE_OPPONENTS = [
    ("src.players.b13902102.best_player1", "BestPlayer1", 2.0),
    ("src.players.b13902102.best_player2", "BestPlayer2", 1.5),
    ("src.players.b13902102.adaptive_portfolio_player", "AdaptivePortfolioPlayer", 1.5),
    ("src.players.b13902102.rollout_search_player", "RolloutSearchPlayer", 3.5),
    ("src.players.b13902102.beam_expectimax_player", "BeamExpectimaxPlayer", 1.0),
    # ("src.players.TA.public_baselines2", "Baseline10", 8.0),
    # ("src.players.TA.public_baselines2", "Baseline9", 4.0),
    # ("src.players.TA.public_baselines2", "Baseline8", 4.0),
    # ("src.players.TA.public_baselines2", "Baseline7", 4.0),
    # ("src.players.TA.public_baselines2", "Baseline6", 4.0),
    # ("src.players.TA.public_baselines1", "Baseline5", 2.0),
    # ("src.players.TA.public_baselines1", "Baseline4", 2.0),
    # ("src.players.TA.public_baselines1", "Baseline3", 1.0),
    # ("src.players.TA.public_baselines1", "Baseline2", 1.0),
    # ("src.players.TA.public_baselines1", "Baseline1", 1.0),
]

BASELINE_OPPONENT_ARGS = {
    # ("src.players.b13902102.best_player1", "BestPlayer1"): {
    #     "deadline_s": 0.20,
    #     "rollout_min": 8,
    #     "rollout_max": 24,
    #     "cand_keep": 6,
    # },
    # ("src.players.b13902102.best_player2", "BestPlayer2"): {
    #     "deadline_s": 0.20,
    #     "rollout_min": 8,
    #     "rollout_max": 24,
    #     "cand_keep": 6,
    # },
    ("src.players.b13902102.adaptive_portfolio_player", "AdaptivePortfolioPlayer"): {
        "deadline_s": 0.50,
        # "samples_min": 16,
        # "samples_max": 32,
        # "cand_keep": 6,
    },
    ("src.players.b13902102.rollout_search_player", "RolloutSearchPlayer"): {
        "deadline_s": 0.90,
    },
    ("src.players.b13902102.beam_expectimax_player", "BeamExpectimaxPlayer"): {
        "deadline_s": 0.20,
    },
}

HISTORY_POOL_SIZE = 10


HYPERPARAMETER_PROFILES = {
    "train": {
        "iterations": 500,
        "rollout_games": 128,
        "minibatch_size": 512,
        "update_epochs": 6,
        "learning_rate": 5.0e-5,
        "final_lr_ratio": 0.10,
        "lr_schedule": "linear",
        "clip_ratio": 0.20,
        "entropy_coef": 0.010,
        "target_kl": 0.020,
        "max_grad_norm": 1.0,
        "policy_temperature": 1.10,
        "history_probability": 0.20,
        "baseline_probability": 0.10,
        "history_snapshot_interval": 5,
        "history_min_age": 5,
        "history_max_age": 50,
        "history_recent_age": 20,
        "history_recent_probability": 0.70,
    },
    "finetune": {
        "iterations": 200,
        "rollout_games": 192,
        "minibatch_size": 512,
        "update_epochs": 4,
        "learning_rate": 1.0e-5,
        "final_lr_ratio": 0.20,
        "lr_schedule": "linear",
        "clip_ratio": 0.10,
        "entropy_coef": 0.003,
        "target_kl": 0.010,
        "max_grad_norm": 0.5,
        "policy_temperature": 1.00,
        "history_probability": 0.20,
        "baseline_probability": 0.05,
        "history_snapshot_interval": 5,
        "history_min_age": 5,
        "history_max_age": 50,
        "history_recent_age": 20,
        "history_recent_probability": 0.75,
    },
}


_WORKER_CURRENT_MODEL = None
_WORKER_HISTORY_MODELS = ()
_WORKER_BASELINE_SPECS = ()
_WORKER_BASELINE_ARGS = {}


def _worker_init(
    current_model,
    history_models,
    baseline_specs,
    baseline_args,
    limit_torch_threads=True,
):
    global _WORKER_CURRENT_MODEL
    global _WORKER_HISTORY_MODELS
    global _WORKER_BASELINE_SPECS
    global _WORKER_BASELINE_ARGS
    if limit_torch_threads:
        torch.set_num_threads(1)
    _WORKER_CURRENT_MODEL = current_model
    _WORKER_CURRENT_MODEL.eval()
    _WORKER_HISTORY_MODELS = tuple(history_models)
    for model in _WORKER_HISTORY_MODELS:
        model.eval()
    _WORKER_BASELINE_SPECS = tuple(baseline_specs)
    _WORKER_BASELINE_ARGS = {
        tuple(key): dict(value)
        for key, value in baseline_args.items()
    }


def _instantiate_baseline(spec, player_idx):
    module_path, class_name, _ = spec
    module = importlib.import_module(module_path)
    player_class = getattr(module, class_name)
    kwargs = _WORKER_BASELINE_ARGS.get((module_path, class_name), {})
    return player_class(player_idx=player_idx, **kwargs)


def _sample_baseline_spec(rng):
    weights = [float(spec[2]) for spec in _WORKER_BASELINE_SPECS]
    return rng.choices(_WORKER_BASELINE_SPECS, weights=weights, k=1)[0]


def _sample_action(
    model,
    hand,
    history,
    seat,
    temperature,
    keep_transition,
    torch_rng,
):
    features, legal_mask = extract_policy_features(hand, history, seat)
    with torch.no_grad():
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        legal_tensor = torch.from_numpy(legal_mask).bool().unsqueeze(0)
        logits = mask_logits(model(features_tensor), legal_tensor).squeeze(0)
        log_probs = torch.log_softmax(logits / max(float(temperature), 1.0e-6), dim=-1)
        probs = torch.exp(log_probs)
        action_idx = int(torch.multinomial(probs, 1, generator=torch_rng).item())
        old_logprob = float(log_probs[action_idx].item())
    card = action_idx + 1
    if keep_transition:
        transition = (features, legal_mask, action_idx, old_logprob, seat)
    else:
        transition = None
    return card, transition


def _collect_episode(task):
    (
        seed,
        opponent_source,
        history_slot,
        policy_temperature,
    ) = task
    rng = random.Random(seed)
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(seed)

    learner_seats = {rng.randrange(4)}
    seat_models = []
    baseline_players = {}
    for seat in range(4):
        if seat in learner_seats or opponent_source == "current":
            seat_models.append(_WORKER_CURRENT_MODEL)
        elif opponent_source == "history":
            seat_models.append(_WORKER_HISTORY_MODELS[int(history_slot)])
        else:
            seat_models.append(None)
            baseline_players[seat] = _instantiate_baseline(
                _sample_baseline_spec(rng),
                seat,
            )

    state = deal_light_game(seed=rng.randrange(10**9))
    pending = []
    while not state.is_terminal():
        history = state.history_for_player()
        actions = []
        for seat, hand in enumerate(state.hands):
            if not hand:
                continue
            if seat in baseline_players:
                card = baseline_players[seat].action(hand, history)
                if not isinstance(card, int) or card not in hand:
                    card = min(hand)
                transition = None
            else:
                card, transition = _sample_action(
                    seat_models[seat],
                    hand,
                    history,
                    seat,
                    policy_temperature,
                    seat in learner_seats,
                    torch_rng,
                )
            actions.append((seat, card))
            if transition is not None:
                pending.append(transition)
        state.step(actions)

    rows = [
        (features, legal_mask, action_idx, old_logprob, state.utility(seat))
        for features, legal_mask, action_idx, old_logprob, seat in pending
    ]
    return rows, opponent_source


def _resolve_path(path):
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def _load_initial_policy(init_mode, checkpoint_path, device):
    if init_mode == "checkpoint":
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {checkpoint_path}"
            )
        model, optimizer_state, metadata = load_checkpoint(
            checkpoint_path,
            device=device,
        )
        return model, optimizer_state, metadata, "checkpoint"
    return PolicyMLP().to(device), None, {}, "random"


def _profile_config(args):
    config = dict(HYPERPARAMETER_PROFILES[args.profile])
    override_names = (
        "iterations",
        "rollout_games",
        "minibatch_size",
        "update_epochs",
        "learning_rate",
        "final_lr_ratio",
        "lr_schedule",
        "clip_ratio",
        "entropy_coef",
        "target_kl",
        "max_grad_norm",
        "policy_temperature",
        "history_probability",
        "baseline_probability",
        "history_snapshot_interval",
        "history_min_age",
        "history_max_age",
        "history_recent_age",
        "history_recent_probability",
    )
    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    return config


def _scheduled_learning_rate(config, iteration):
    initial_lr = float(config["learning_rate"])
    final_lr = initial_lr * float(config["final_lr_ratio"])
    total = max(1, int(config["iterations"]) - 1)
    progress = min(1.0, max(0.0, (int(iteration) - 1) / total))
    schedule = config["lr_schedule"]
    if schedule == "constant":
        return initial_lr
    if schedule == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (initial_lr - final_lr) * weight
    return initial_lr + (final_lr - initial_lr) * progress


def _make_shared_policy():
    model = PolicyMLP().cpu()
    model.share_memory()
    model.eval()
    return model


def _copy_model_weights(source, destination):
    source_state = source.state_dict()
    destination_state = destination.state_dict()
    with torch.no_grad():
        for name, destination_tensor in destination_state.items():
            destination_tensor.copy_(source_state[name].detach().cpu())
    destination.eval()


def _store_history_snapshot(model, history_models, history_entries, iteration):
    slot = 0
    used_slots = {entry["slot"] for entry in history_entries}
    if len(history_entries) < HISTORY_POOL_SIZE:
        for candidate in range(HISTORY_POOL_SIZE):
            if candidate not in used_slots:
                slot = candidate
                break
    else:
        oldest = min(history_entries, key=lambda entry: entry["iteration"])
        slot = oldest["slot"]
        history_entries.remove(oldest)
    _copy_model_weights(model, history_models[slot])
    history_entries.append({"slot": slot, "iteration": int(iteration)})


def _eligible_history(history_entries, iteration, config):
    eligible = []
    for entry in history_entries:
        age = int(iteration) - int(entry["iteration"])
        if int(config["history_min_age"]) <= age <= int(config["history_max_age"]):
            eligible.append({**entry, "age": age})
    return eligible


def _sample_history_slot(rng, eligible, config):
    recent = [
        entry
        for entry in eligible
        if entry["age"] <= int(config["history_recent_age"])
    ]
    if recent and rng.random() < float(config["history_recent_probability"]):
        return rng.choice(recent)["slot"]
    return rng.choice(eligible)["slot"]


def _sample_opponent_source(rng, eligible_history, config, baselines_enabled):
    history_probability = (
        float(config["history_probability"])
        if eligible_history
        else 0.0
    )
    baseline_probability = (
        float(config["baseline_probability"])
        if baselines_enabled
        else 0.0
    )
    roll = rng.random()
    if roll < history_probability:
        return "history", _sample_history_slot(rng, eligible_history, config)
    if roll < history_probability + baseline_probability:
        return "baseline", -1
    return "current", -1


def _collect_rollout(
    pool,
    num_workers,
    rollout_games,
    rng,
    iteration,
    history_entries,
    config,
    baselines_enabled,
    policy_temperature,
):
    eligible_history = _eligible_history(history_entries, iteration, config)
    tasks = []
    for _ in range(int(rollout_games)):
        opponent_source, history_slot = _sample_opponent_source(
            rng,
            eligible_history,
            config,
            baselines_enabled,
        )
        tasks.append(
            (
                rng.randrange(2**31),
                opponent_source,
                history_slot,
                float(policy_temperature),
            )
        )
    if num_workers > 1:
        results = pool.imap_unordered(_collect_episode, tasks, chunksize=1)
    else:
        results = map(_collect_episode, tasks)

    rows = []
    source_counts = {"current": 0, "history": 0, "baseline": 0}
    progress = tqdm(
        results,
        total=len(tasks),
        desc=f"PPO rollout {iteration}",
        unit="game",
        leave=False,
    )
    for episode_rows, opponent_source in progress:
        rows.extend(episode_rows)
        source_counts[opponent_source] += 1
        progress.set_postfix(
            transitions=len(rows),
            current=source_counts["current"],
            history=source_counts["history"],
            baseline=source_counts["baseline"],
        )
    return rows, source_counts, len(eligible_history)


def _make_rollout_tensors(rows, device):
    features = torch.from_numpy(np.stack([row[0] for row in rows])).float().to(device)
    legal_masks = torch.from_numpy(np.stack([row[1] for row in rows])).bool().to(device)
    actions = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([row[3] for row in rows], dtype=torch.float32, device=device)
    returns = torch.tensor([row[4] for row in rows], dtype=torch.float32, device=device)
    advantages = returns - returns.mean()
    advantages = advantages / returns.std(unbiased=False).clamp_min(1.0e-6)
    return features, legal_masks, actions, old_log_probs, returns, advantages


def _ppo_update(model, optimizer, rows, config, device, iteration):
    features, legal_masks, actions, old_log_probs, returns, advantages = _make_rollout_tensors(
        rows, device
    )
    num_samples = actions.shape[0]
    minibatch_size = min(int(config["minibatch_size"]), num_samples)
    target_kl = float(config["target_kl"])
    temperature = max(float(config["policy_temperature"]), 1.0e-6)
    metrics = {
        "loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "batches": 0,
        "epochs": 0,
        "mean_return": float(returns.mean().item()),
    }

    model.train()
    epoch_bar = tqdm(
        range(1, int(config["update_epochs"]) + 1),
        desc=f"PPO update {iteration}",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_bar:
        permutation = torch.randperm(num_samples, device=device)
        epoch_kl_sum = 0.0
        epoch_samples = 0
        for start in range(0, num_samples, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            logits = mask_logits(model(features[indices]), legal_masks[indices])
            log_probs = torch.log_softmax(logits / temperature, dim=-1)
            chosen_log_probs = log_probs.gather(1, actions[indices].unsqueeze(1)).squeeze(1)
            log_ratio = chosen_log_probs - old_log_probs[indices]
            ratio = torch.exp(log_ratio)
            batch_advantages = advantages[indices]
            unclipped = ratio * batch_advantages
            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - float(config["clip_ratio"]),
                    1.0 + float(config["clip_ratio"]),
                )
                * batch_advantages
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = policy_loss - float(config["entropy_coef"]) * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["max_grad_norm"])
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > float(config["clip_ratio"])
                ).float().mean()
            batch_count = int(indices.shape[0])
            epoch_kl_sum += float(approx_kl.item()) * batch_count
            epoch_samples += batch_count
            metrics["loss"] += float(loss.detach().item()) * batch_count
            metrics["entropy"] += float(entropy.detach().item()) * batch_count
            metrics["approx_kl"] += float(approx_kl.item()) * batch_count
            metrics["clip_fraction"] += float(clip_fraction.item()) * batch_count
            metrics["batches"] += batch_count

        metrics["epochs"] = epoch
        epoch_kl = epoch_kl_sum / max(1, epoch_samples)
        epoch_bar.set_postfix(kl=f"{epoch_kl:.4f}")
        if target_kl > 0.0 and epoch_kl > target_kl:
            tqdm.write(
                f"[PPO] Iteration {iteration}: early stop after epoch {epoch}; "
                f"KL {epoch_kl:.4f} > target {target_kl:.4f}"
            )
            break

    denominator = max(1, metrics.pop("batches"))
    for name in ("loss", "entropy", "approx_kl", "clip_fraction"):
        metrics[name] /= denominator
    return metrics


def train_ppo_onpolicy(args):
    config = _profile_config(args)
    if args.disable_history:
        config["history_probability"] = 0.0
    baselines_enabled = bool(BASELINE_OPPONENTS) and not args.disable_baselines
    if not baselines_enabled:
        config["baseline_probability"] = 0.0
    if int(config["iterations"]) < 1:
        raise ValueError("iterations must be at least 1")
    if int(config["rollout_games"]) < 1:
        raise ValueError("rollout_games must be at least 1")
    if int(config["minibatch_size"]) < 1:
        raise ValueError("minibatch_size must be at least 1")
    if int(config["update_epochs"]) < 1:
        raise ValueError("update_epochs must be at least 1")
    if int(config["history_snapshot_interval"]) < 1:
        raise ValueError("history_snapshot_interval must be at least 1")
    if int(config["history_min_age"]) < 0:
        raise ValueError("history_min_age must be non-negative")
    if int(config["history_max_age"]) < int(config["history_min_age"]):
        raise ValueError("history_max_age must be at least history_min_age")
    probability_names = (
        "history_probability",
        "baseline_probability",
        "history_recent_probability",
    )
    for name in probability_names:
        if not 0.0 <= float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if (
        float(config["history_probability"])
        + float(config["baseline_probability"])
        > 1.0
    ):
        raise ValueError(
            "history_probability + baseline_probability must not exceed 1"
        )
    init_checkpoint = _resolve_path(args.init_checkpoint)
    output_checkpoint = _resolve_path(args.output_checkpoint)

    print(f"[PPO] Profile: {args.profile}", flush=True)
    print(f"[PPO] Output checkpoint: {output_checkpoint}", flush=True)
    print(f"[PPO] Workers: {args.num_workers}", flush=True)
    print(f"[PPO] Hyperparameters: {config}", flush=True)
    print(
        f"[PPO] Historical pool: memory-only, capacity={HISTORY_POOL_SIZE}",
        flush=True,
    )
    print(
        f"[PPO] Baseline pool ({len(BASELINE_OPPONENTS) if baselines_enabled else 0}): "
        f"{[spec[:2] for spec in BASELINE_OPPONENTS] if baselines_enabled else []}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, optimizer_state, _, initialization = _load_initial_policy(
        args.init_mode,
        init_checkpoint,
        args.device,
    )
    if initialization == "checkpoint":
        print(f"[PPO] Initial checkpoint: {init_checkpoint}", flush=True)
    else:
        print("[PPO] Creating a fresh random policy", flush=True)
    init_checkpoint_metadata = (
        str(init_checkpoint)
        if initialization == "checkpoint"
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=1.0e-4,
    )
    if args.resume_optimizer and optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    rollout_model = _make_shared_policy()
    history_models = [_make_shared_policy() for _ in range(HISTORY_POOL_SIZE)]
    _copy_model_weights(model, rollout_model)
    for history_model in history_models:
        _copy_model_weights(model, history_model)
    history_entries = []

    context = mp.get_context("spawn")
    pool = None
    if args.num_workers > 1:
        pool = context.Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=(
                rollout_model,
                history_models,
                BASELINE_OPPONENTS if baselines_enabled else (),
                BASELINE_OPPONENT_ARGS if baselines_enabled else {},
                True,
            ),
        )
    else:
        _worker_init(
            rollout_model,
            history_models,
            BASELINE_OPPONENTS if baselines_enabled else (),
            BASELINE_OPPONENT_ARGS if baselines_enabled else {},
            False,
        )

    total_games = 0
    last_saved_iteration = 0
    try:
        iterations = tqdm(
            range(1, int(config["iterations"]) + 1),
            desc="PPO on-policy",
            unit="iter",
        )
        for iteration in iterations:
            learning_rate = _scheduled_learning_rate(config, iteration)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            _copy_model_weights(model, rollout_model)
            if (
                (iteration - 1) % int(config["history_snapshot_interval"])
                == 0
            ):
                _store_history_snapshot(
                    rollout_model,
                    history_models,
                    history_entries,
                    iteration,
                )

            model.eval()
            rows, source_counts, eligible_history = _collect_rollout(
                pool=pool,
                num_workers=args.num_workers,
                rollout_games=config["rollout_games"],
                rng=rng,
                iteration=iteration,
                history_entries=history_entries,
                config=config,
                baselines_enabled=baselines_enabled,
                policy_temperature=config["policy_temperature"],
            )
            if not rows:
                raise RuntimeError("Rollout collection returned no learner transitions")

            total_games += int(config["rollout_games"])
            metrics = _ppo_update(
                model=model,
                optimizer=optimizer,
                rows=rows,
                config=config,
                device=args.device,
                iteration=iteration,
            )
            iterations.set_postfix(
                lr=f"{learning_rate:.2e}",
                ret=f"{metrics['mean_return']:.3f}",
                kl=f"{metrics['approx_kl']:.4f}",
                clip=f"{metrics['clip_fraction']:.3f}",
            )
            tqdm.write(
                f"[PPO] Iteration {iteration}/{config['iterations']}: "
                f"games={config['rollout_games']}, transitions={len(rows)}, "
                f"sources={source_counts}, history_ready={eligible_history}, "
                f"epochs={metrics['epochs']}, return={metrics['mean_return']:.4f}, "
                f"loss={metrics['loss']:.4f}, entropy={metrics['entropy']:.4f}, "
                f"kl={metrics['approx_kl']:.4f}, "
                f"clip_fraction={metrics['clip_fraction']:.4f}"
            )

            if iteration % max(1, args.save_every) == 0:
                save_checkpoint(
                    output_checkpoint,
                    model,
                    optimizer,
                    stage="ppo_onpolicy",
                    profile=args.profile,
                    iteration=iteration,
                    total_games=total_games,
                    hyperparameters=config,
                    initialization=initialization,
                    init_checkpoint=init_checkpoint_metadata,
                    history_pool_size=HISTORY_POOL_SIZE,
                    baseline_opponents=BASELINE_OPPONENTS,
                )
                last_saved_iteration = iteration
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if last_saved_iteration != int(config["iterations"]):
        save_checkpoint(
            output_checkpoint,
            model,
            optimizer,
            stage="ppo_onpolicy",
            profile=args.profile,
            iteration=int(config["iterations"]),
            total_games=total_games,
            hyperparameters=config,
            initialization=initialization,
            init_checkpoint=init_checkpoint_metadata,
            history_pool_size=HISTORY_POOL_SIZE,
            baseline_opponents=BASELINE_OPPONENTS,
        )
    print(f"[PPO] Saved final checkpoint: {output_checkpoint}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="On-policy actor-only PPO self-play trainer."
    )
    parser.add_argument(
        "--init-mode",
        choices=("checkpoint", "random"),
        default="checkpoint",
        help="Load a checkpoint or create a fresh random policy.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="artifacts/b13902102/bc_policy.pth",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        default="artifacts/b13902102/ppo_onpolicy.pth",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(HYPERPARAMETER_PROFILES),
        default="train",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--disable-history", action="store_true")
    parser.add_argument("--disable-baselines", action="store_true")

    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--rollout-games", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--final-lr-ratio", type=float, default=None)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear", "cosine"),
        default=None,
    )
    parser.add_argument("--clip-ratio", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--policy-temperature", type=float, default=None)
    parser.add_argument("--history-probability", type=float, default=None)
    parser.add_argument("--baseline-probability", type=float, default=None)
    parser.add_argument("--history-snapshot-interval", type=int, default=None)
    parser.add_argument("--history-min-age", type=int, default=None)
    parser.add_argument("--history-max-age", type=int, default=None)
    parser.add_argument("--history-recent-age", type=int, default=None)
    parser.add_argument("--history-recent-probability", type=float, default=None)
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    train_ppo_onpolicy(args)


if __name__ == "__main__":
    main()
