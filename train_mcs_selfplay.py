import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from src.players.b13902102.rl_checkpoint import save_checkpoint
from src.players.b13902102.rl_model import PolicyMLP
from src.players.b13902102.rl_population import PopulationTrainer


def ensure_initial_checkpoint(path):
    path = Path(path)
    if not path.exists():
        print(f"[MCS] Initial checkpoint not found; creating random policy: {path}", flush=True)
        model = PolicyMLP()
        save_checkpoint(path, model, stage="random_init")
    return path


def standings_line(trainer):
    ordered = sorted(trainer.slots, key=lambda slot: slot.ema_points, reverse=True)
    return " | ".join(
        f"A{slot.agent_id}:pts={slot.ema_points:.2f},rank={slot.ema_rank:.2f},elo={slot.elo:.0f}"
        for slot in ordered
    )


def main():
    parser = argparse.ArgumentParser(description="Alpha0.5/MCS population self-play trainer.")
    parser.add_argument("--init-checkpoint", type=str, default="artifacts/b13902102/bc_policy.pth")
    parser.add_argument("--population-dir", type=str, default="artifacts/b13902102/population")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--games-per-generation", type=int, default=192)
    parser.add_argument("--train-sims", type=int, default=32)
    parser.add_argument("--fast-eval-partitions", type=int, default=12)
    parser.add_argument("--full-eval-period", type=int, default=4)
    parser.add_argument("--full-eval-partitions", type=int, default=24)
    parser.add_argument("--eval-sims", type=int, default=8)
    parser.add_argument("--replace-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("[MCS] Preparing population self-play", flush=True)
    init_ckpt = ensure_initial_checkpoint(args.init_checkpoint)
    trainer = PopulationTrainer(args.population_dir, seed=args.seed)
    print(f"[MCS] Initializing 8-agent population from: {init_ckpt}", flush=True)
    trainer.initialize_from_checkpoint(init_ckpt)

    generations = tqdm(range(1, args.generations + 1), desc="MCS generations", unit="gen")
    for generation in generations:
        tqdm.write(f"[MCS] Generation {generation}/{args.generations}: self-play training")
        trainer.train_selfplay_games(
            args.games_per_generation,
            sims=args.train_sims,
            progress=True,
            desc=f"Gen {generation} self-play",
        )
        tqdm.write(f"[MCS] Generation {generation}/{args.generations}: fast frozen eval")
        fast_records = trainer.evaluate_frozen(
            partitions=args.fast_eval_partitions,
            sims=max(1, args.eval_sims // 2),
            progress=True,
            desc=f"Gen {generation} fast eval",
        )
        generations.set_postfix(best=max(slot.ema_points for slot in trainer.slots), worst=min(slot.ema_points for slot in trainer.slots))
        tqdm.write(f"[MCS] Fast eval done: {standings_line(trainer)}")
        if generation % args.full_eval_period == 0:
            tqdm.write(f"[MCS] Generation {generation}/{args.generations}: full frozen eval")
            trainer.evaluate_frozen(
                partitions=args.full_eval_partitions,
                sims=args.eval_sims,
                progress=True,
                desc=f"Gen {generation} full eval",
            )
            replacements = trainer.replace_bottom(generation, replace_count=args.replace_count)
            archived = trainer.save_archive(generation)
            tqdm.write(f"[MCS] Full eval done: {standings_line(trainer)}")
            tqdm.write(f"[MCS] Replacements: {replacements if replacements else 'none'}")
            tqdm.write(f"[MCS] Archived checkpoints: {[str(path) for path in archived]}")
    print("[MCS] Done", flush=True)


if __name__ == "__main__":
    main()
