import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.players.b13902102.rl_checkpoint import save_checkpoint
from src.players.b13902102.rl_model import PolicyMLP
from src.players.b13902102.rl_population import PopulationTrainer


def ensure_initial_checkpoint(path):
    path = Path(path)
    if not path.exists():
        model = PolicyMLP()
        save_checkpoint(path, model, stage="random_init")
    return path


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

    init_ckpt = ensure_initial_checkpoint(args.init_checkpoint)
    trainer = PopulationTrainer(args.population_dir, seed=args.seed)
    trainer.initialize_from_checkpoint(init_ckpt)

    for generation in range(1, args.generations + 1):
        trainer.train_selfplay_games(args.games_per_generation, sims=args.train_sims)
        trainer.evaluate_frozen(partitions=args.fast_eval_partitions, sims=max(1, args.eval_sims // 2))
        if generation % args.full_eval_period == 0:
            trainer.evaluate_frozen(partitions=args.full_eval_partitions, sims=args.eval_sims)
            trainer.replace_bottom(generation, replace_count=args.replace_count)
            trainer.save_archive(generation)


if __name__ == "__main__":
    main()
