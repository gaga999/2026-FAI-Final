import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.players.b13902102.rl_population_memory import MemoryPopulationTrainer


POOL_CHUNKSIZE = 1
ROOT_EXPLORATION_MIX = 0.03
TRAIN_HEURISTIC_MIX = 0.40
TRAIN_RANDOM_MIX = 0.10
EVAL_HEURISTIC_MIX = 0.25
EVAL_RANDOM_MIX = 0.10


def validate_static_config():
    if POOL_CHUNKSIZE < 1:
        raise ValueError("POOL_CHUNKSIZE must be at least 1")
    for name, value in (
        ("ROOT_EXPLORATION_MIX", ROOT_EXPLORATION_MIX),
        ("TRAIN_HEURISTIC_MIX", TRAIN_HEURISTIC_MIX),
        ("TRAIN_RANDOM_MIX", TRAIN_RANDOM_MIX),
        ("EVAL_HEURISTIC_MIX", EVAL_HEURISTIC_MIX),
        ("EVAL_RANDOM_MIX", EVAL_RANDOM_MIX),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if TRAIN_HEURISTIC_MIX + TRAIN_RANDOM_MIX > 1.0:
        raise ValueError("Training rollout probabilities must not exceed 1")
    if EVAL_HEURISTIC_MIX + EVAL_RANDOM_MIX > 1.0:
        raise ValueError("Evaluation rollout probabilities must not exceed 1")


def scheduled_learning_rate(initial_lr, final_lr_ratio, schedule, generation, total):
    progress = (int(generation) - 1) / max(1, int(total) - 1)
    final_lr = float(initial_lr) * float(final_lr_ratio)
    if schedule == "linear":
        return float(initial_lr) + (final_lr - float(initial_lr)) * progress
    if schedule == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (float(initial_lr) - final_lr) * weight
    return float(initial_lr)


def standings_line(trainer):
    ordered = sorted(
        trainer.slots,
        key=lambda slot: slot.ema_points,
        reverse=True,
    )
    return " | ".join(
        f"A{slot.agent_id}:pts={slot.ema_points:.2f},"
        f"rank={slot.ema_rank:.2f},elo={slot.elo:.0f}"
        for slot in ordered
    )


def train(args):
    validate_static_config()
    if args.num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if args.games_per_generation < 1:
        raise ValueError("games_per_generation must be at least 1")
    if args.replay_size < args.batch_size:
        raise ValueError("replay_size must be at least batch_size")

    population_dir = Path(args.population_dir)
    if not population_dir.is_absolute():
        population_dir = ROOT / population_dir
    torch.manual_seed(args.seed)
    trainer = MemoryPopulationTrainer(
        population_dir=population_dir,
        device=args.device,
        seed=args.seed,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        replay_limit=args.replay_size,
        entropy_coef=args.entropy_coef,
    )
    if args.resume:
        trainer.resume()
        start_generation = trainer.generation + 1
        print(
            f"[MCS memory] Resumed generation={trainer.generation}, "
            f"games={trainer.total_games}",
            flush=True,
        )
    else:
        if args.init_mode == "checkpoint":
            init_checkpoint = Path(args.init_checkpoint)
            if not init_checkpoint.is_absolute():
                init_checkpoint = ROOT / init_checkpoint
            trainer.initialize_from_checkpoint(init_checkpoint)
            initialization_description = str(init_checkpoint)
        else:
            trainer.initialize_random()
            initialization_description = "fresh random models"
        trainer.save_population(0, reason="initialization")
        start_generation = 1
        print(
            f"[MCS memory] Initialized 8 agents from: "
            f"{initialization_description}",
            flush=True,
        )

    print(
        f"[MCS memory] workers={args.num_workers}, replay_per_agent={args.replay_size}, "
        f"batch={args.batch_size}, entropy={args.entropy_coef}, "
        f"save_every_games={args.save_every_games}, chunksize={POOL_CHUNKSIZE}, "
        f"train_rollout=(heuristic={TRAIN_HEURISTIC_MIX}, "
        f"random={TRAIN_RANDOM_MIX}, "
        f"model={1.0 - TRAIN_HEURISTIC_MIX - TRAIN_RANDOM_MIX}), "
        f"root_exploration={ROOT_EXPLORATION_MIX}",
        flush=True,
    )
    next_periodic_save = (
        (trainer.total_games // args.save_every_games) + 1
    ) * args.save_every_games

    generations = tqdm(
        range(start_generation, args.generations + 1),
        desc="MCS memory generations",
        unit="gen",
    )
    try:
        for generation in generations:
            learning_rate = scheduled_learning_rate(
                args.learning_rate,
                args.final_lr_ratio,
                args.lr_schedule,
                generation,
                args.generations,
            )
            trainer.set_learning_rate(learning_rate)
            trainer.collect_training_games(
                args.games_per_generation,
                sims=args.train_sims,
                num_workers=args.num_workers,
                exploration_mix=ROOT_EXPLORATION_MIX,
                heuristic_mix=TRAIN_HEURISTIC_MIX,
                random_mix=TRAIN_RANDOM_MIX,
                task_chunksize=POOL_CHUNKSIZE,
                search_deadline_s=(
                    None
                    if args.search_deadline <= 0.0
                    else args.search_deadline
                ),
                progress=True,
                desc=f"Gen {generation} collect",
            )
            update_metrics = trainer.update_population(
                update_steps_per_agent=args.updates_per_agent,
                progress=True,
                desc=f"Gen {generation} update",
            )
            mean_loss = sum(
                metrics["loss"]
                for metrics in update_metrics.values()
                if metrics["updates"] > 0
            ) / max(
                1,
                sum(
                    1
                    for metrics in update_metrics.values()
                    if metrics["updates"] > 0
                ),
            )

            trainer.evaluate_frozen(
                partitions=args.fast_eval_partitions,
                sims=max(1, args.eval_sims // 2),
                num_workers=args.num_workers,
                heuristic_mix=EVAL_HEURISTIC_MIX,
                random_mix=EVAL_RANDOM_MIX,
                task_chunksize=POOL_CHUNKSIZE,
                search_deadline_s=(
                    None
                    if args.search_deadline <= 0.0
                    else args.search_deadline
                ),
                progress=True,
                desc=f"Gen {generation} fast eval",
            )
            generations.set_postfix(
                games=trainer.total_games,
                lr=f"{learning_rate:.2e}",
                loss=f"{mean_loss:.4f}",
                best=f"{max(slot.ema_points for slot in trainer.slots):.2f}",
            )
            tqdm.write(
                f"[MCS memory] Generation {generation}: "
                f"games={trainer.total_games}, lr={learning_rate:.2e}, "
                f"loss={mean_loss:.4f} | {standings_line(trainer)}"
            )

            if trainer.total_games >= next_periodic_save:
                trainer.save_population(generation, reason="game_interval")
                while next_periodic_save <= trainer.total_games:
                    next_periodic_save += args.save_every_games
                tqdm.write(
                    f"[MCS memory] Periodic checkpoint saved at "
                    f"{trainer.total_games} games"
                )

            if generation % args.full_eval_period == 0:
                trainer.evaluate_frozen(
                    partitions=args.full_eval_partitions,
                    sims=args.eval_sims,
                    num_workers=args.num_workers,
                    heuristic_mix=EVAL_HEURISTIC_MIX,
                    random_mix=EVAL_RANDOM_MIX,
                    task_chunksize=POOL_CHUNKSIZE,
                    search_deadline_s=(
                        None
                        if args.search_deadline <= 0.0
                        else args.search_deadline
                    ),
                    progress=True,
                    desc=f"Gen {generation} full eval",
                )
                replacements = trainer.replace_bottom(
                    generation,
                    replace_count=args.replace_count,
                )
                trainer.save_population(generation, reason="replacement")
                archived = trainer.save_archive(generation)
                tqdm.write(
                    f"[MCS memory] Full eval: {standings_line(trainer)}"
                )
                tqdm.write(
                    f"[MCS memory] Replacements: "
                    f"{replacements if replacements else 'none'}"
                )
                tqdm.write(
                    f"[MCS memory] Archive: {[str(path) for path in archived]}"
                )

        if args.generations % args.full_eval_period != 0:
            trainer.save_population(args.generations, reason="final")
            trainer.save_archive(args.generations)
    finally:
        trainer.close_workers()
    print("[MCS memory] Done", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-first parallel Alpha0.5/MCS population trainer."
    )
    parser.add_argument(
        "--init-mode",
        choices=("checkpoint", "random"),
        default="checkpoint",
        help="Load one checkpoint into the population or create fresh random models.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="artifacts/b13902102/bc_policy.pth",
    )
    parser.add_argument(
        "--population-dir",
        type=str,
        default="artifacts/b13902102/population_memory",
    )
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--games-per-generation", type=int, default=192)
    parser.add_argument("--train-sims", type=int, default=32)
    parser.add_argument(
        "--updates-per-agent",
        type=int,
        default=96,
        help="Default matches the original trainer's approximate samples updated per generation.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear", "cosine"),
        default="constant",
    )
    parser.add_argument("--final-lr-ratio", type=float, default=0.30)
    parser.add_argument("--fast-eval-partitions", type=int, default=12)
    parser.add_argument("--full-eval-period", type=int, default=4)
    parser.add_argument("--full-eval-partitions", type=int, default=24)
    parser.add_argument("--eval-sims", type=int, default=8)
    parser.add_argument("--replace-count", type=int, default=1)
    parser.add_argument("--save-every-games", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--search-deadline",
        type=float,
        default=0.2,
        help="Per-decision MCS deadline; use 0 to disable the deadline.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.save_every_games < 1:
        parser.error("--save-every-games must be at least 1")
    if args.full_eval_period < 1:
        parser.error("--full-eval-period must be at least 1")
    train(args)


if __name__ == "__main__":
    main()
