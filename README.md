# 2026 FAI Final - Building a 6 Nimmt! Agent 🐮

## Setup

Use `python 3.13.11` and `pip install -r requirements.txt`.

## Running Simulations and Tournaments

The framework provides two main scripts for evaluating players, both driven by JSON configuration files (examples located in `configs/game/` and `configs/tournament/`).

### 1. Single Game Simulation
Use `run_single_game.py` to run a single match. It supports detailed logging and captures initial hands and history outputs.
```bash
python run_single_game.py --config configs/game/example.json
```

### 2. Tournaments
Use `run_tournament.py` to run large-scale evaluations.
```bash
python run_tournament.py --config configs/tournament/example.json
```
*Optional config overrides*: You can mix-and-match configurations via the command line using `--player-cfg`, `--engine-cfg`, or `--tournament-cfg`.

## Config File Structure

Configuration files are structured as JSON objects with three main sections:

### 1. `players`
A list of players participating in the game or tournament. 
*   **Format**: Defined as either a dictionary `{"path": "...", "class": "...", "args": {...}, "label": "optional"}` or a compact list `["path", "class", {"args": "here"}, "optional_label"]`.
*   *Note: The compact list can be just length-2 `["path", "class"]` or 3 if no additional arguments or labels are needed. The label helps distinguish players using the same class.*

### 1b. `baselines`
An optional list with the same format as `players`. In `random_partition`-style tournaments, baseline entrants are scheduled exactly like normal players, but can also be used as score anchors for the final standings.

### 2. `engine`
Settings that control the inner game mechanics:
*   `n_players`: Number of players per game (default: 4).
*   `n_rounds`: Number of rounds played per game (default: 10).
*   `timeout` & `timeout_buffer`: Time limit (in seconds) allowed for a player's `action()`. If they exceed this, their card defaults. If they intentionally catch and swallow the alarm exception, they are disqualified.
*   `verbose`: Boolean to toggle detailed turn-by-turn print logs.

### 3. `tournament`
Settings specific to the `run_tournament.py` runner (ignored by `run_single_game.py`):
*   `grading_mode`: Boolean (default: `false`). When `true`, pins `torch.set_num_threads(1)` (and related env vars like `OMP_NUM_THREADS`) so accidental thread-level parallelism is disabled, and suppresses per-game stdout so only the final standings are printed. This is the mode used during official evaluation.
*   `type`: Type of tournament to run (`combination`, `random_partition`, or `grouped_random_partition`). **The `random_partition` tournament format will be used in final evaluations.**
*   `duplication_mode`: String (`"permutations"`, `"cycle"`, or `"none"`). Determines how hands are duplicated to reduce RNG variance. `"permutations"` plays $N!$ games with all seat assignments, `"cycle"` plays $N$ games shifting hands, and `"none"` plays 1 game. (Legacy boolean `use_permutations` is also supported).
*   `num_games_per_player` & `num_workers`: Used by `random_partition` to control the number of games played and parallel processing threads.
*   `scoring`: (Optional, used by `random_partition`). Adds a calibrated `Score` column in final standings by mapping `avg_rank` to a linear score scale defined by baseline percentiles. Supported keys are `baseline_upper_pct`, `baseline_lower_pct`, `score_at_upper_pct`, and `score_at_lower_pct`.
*   `num_groups`: (Optional, used by `grouped_random_partition`). The number of groups to split the players into for Stage 2 of the tournament based on their Stage 1 rank.
*   `max_memory_mb_per_matchup`: (Optional) Maximum memory limit in MB for a single matchup process to prevent memory leaks or out-of-memory issues from crashing the whole tournament.
*   `matchup_timeout_multiplier`: (Optional) Multiplier applied to the total expected matchup time to determine the hard timeout limit for the matchup subprocess.

## Player Disqualifications and Penalties
During games and tournaments, players may experience errors or consume excessive resources. These are marked in the final standings as:
*   **DQ (Disqualified)**: Player swallowed a timeout exception. Their subsequent moves defaults to their smallest available card.
*   **TO (Timeout)**: Player exceeded the `timeout` limit for a single turn. Their card defaults to their smallest available card.
*   **EXC (Exception)**: Player code raised an exception during their turn. Their card defaults to their smallest available card.
*   **OOM (Out of Memory)**: Player's matchup subprocess was killed due to exceeding the `max_memory_mb_per_matchup` limit. The entire matchup is aborted and players in that matchup receive no points or ranking.
*   **ERR (Generic Error)**: Player's matchup subprocess crashed for reasons other than OOM (e.g. fatal segfault, total process timeout). The entire matchup is aborted and players in that matchup receive no points or ranking.

## ⚠️ Important Warnings for Students
- **Do not use `multiprocessing` or `threading`!** Doing so will cause severe server instability and you will be penalized. The tournament orchestrator already runs games in parallel processes. Your agent must remain single-threaded.
- **Do NOT use bare `except:` or catch `BaseException` in your code.** You may accidentally catch system signals, timeout interrupts (`TimeoutException`), or out-of-memory errors (`MemoryError`), masking critical engine behaviors. Only catch specific exceptions you anticipate (e.g. `except ValueError:`).
- **Penalties for Errors:** High error rates or causing errors intentionally will result in score deductions.

## How to Add New Players
1.  Create a **subdirectory** under `src/players/` (e.g., `src/players/student_id(lowercase)/`).
2.  Add your player Python file(s) inside that directory.
3.  Your player class must implement the `action` method.

**Example:**
File: `src/players/student_id(lowercase)/best_player1.py`
```python
class BestPlayer1:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        # hand: list of integers (your cards)
        # history: dict containing board state and past moves
        return hand[0] # Your logic here
```

## Game Engine Rules
*   **6th Card Rule**: If a row has 5 cards, the 6th card placed takes the row.
*   **Low Card Rule**: If a played card is lower than the last card of all rows, the player takes the row with the **lowest score**.
    *   Tie-breaking: Lowest card count -> Smallest row index.

## Constraints
*   1 second timeout for each round
*   1 GB RAM
*   No multi-process/thread

## b13902102 Agents

The original `best_player1.py` and `best_player2.py` files are kept as legacy players. The new non-training methods are added as separate method-named agents under `src/players/b13902102/`.

* `RiskHeuristicPlayer` in `risk_heuristic_player.py`: fast deterministic risk heuristic. It simulates exact immediate placement, estimates opponent interval pressure, penalizes low-card forced takes and high bullhead cards, and uses a stable fallback to the smallest legal card on ordinary exceptions.
* `RolloutSearchPlayer` in `rollout_search_player.py`: bounded rollout search. It uses the heuristic selector as the fallback, samples hidden opponent hands from unseen cards, resolves turns with a local engine-compatible simulator, and stops before the 1-second timeout.
* `BeamExpectimaxPlayer` in `beam_expectimax_player.py`: deterministic shallow beam/expectimax alternative. It samples plausible opponent actions from heuristic top-k choices and evaluates short-horizon leaf states without training a model.
* `AdaptivePortfolioPlayer` in `adaptive_portfolio_player.py`: one-turn portfolio search designed after comparing the legacy best players. It keeps many cheap current-turn samples, uses an opponent policy mixture, and blends mean cost with worst-case and bad-event rates.

Shared helper files:

* `rules.py`: bullhead scoring, low-card tie-breaks, one-card placement, simultaneous turn resolution, and history parsing.
* `features.py`: stable per-action feature extraction for later model-based agents.
* `heuristic.py`: scoring and card ranking used by all three method agents.
* `rollout.py`: hidden-hand sampling and rollout simulation for `RolloutSearchPlayer`.
* `beam_search.py`: shallow beam/expectimax expansion for `BeamExpectimaxPlayer`.

The implementation follows the uploaded strategy specs where they match the engine. One practical adjustment is that the search players use conservative soft deadlines around `0.70-0.74` seconds rather than trying to consume the whole 1-second action limit, because the official timeout is implemented with a `BaseException` signal and should not be caught.

## Useful b13902102 Configs

Single-game configs:

```bash
python3 run_single_game.py --config configs/game/test.json
```

Tournament configs:

```bash
python3 run_tournament.py --config configs/tournament/test.json
```

`configs/game/test.json` runs the risk heuristic against released public baselines. `configs/tournament/test.json` includes the method agents plus released public baselines as scoring anchors.

## b13902102 RL Training Pipeline

The model-based training scaffold is documented in [docs/b13902102_rl_training_spec.md](/Users/chengjieyuan/Desktop/114/114下/FAI/final/2026-FAI-Final/docs/b13902102_rl_training_spec.md).

Root training entry points:

```bash
python3 train_bc.py --num-games 2000 --checkpoint artifacts/b13902102/bc_policy.pth
python3 train_mcs_selfplay.py --init-mode checkpoint --init-checkpoint artifacts/b13902102/bc_policy.pth --population-dir artifacts/b13902102/population
python3 train_mcs_selfplay_memory.py --init-mode checkpoint --init-checkpoint artifacts/b13902102/bc_policy.pth --num-workers 1
python3 train_ppo_selfplay.py --init-mode checkpoint --init-checkpoint artifacts/b13902102/population/archive/best.pth --output-checkpoint artifacts/b13902102/ppo_policy.pth
python3 train_ppo_onpolicy.py --profile train --init-mode checkpoint --init-checkpoint artifacts/b13902102/bc_policy.pth --num-workers 1
```

`train_mcs_selfplay_memory.py` is the preferred MCS trainer for longer runs. It keeps all eight models and optimizers in the main process, uses a recent 6000-row replay buffer per agent, and writes checkpoints only after `--save-every-games`, replacement/archive events, and final shutdown. `--num-workers N` uses persistent spawned CPU workers and shared frozen model copies; model updates remain in the main process. `POOL_CHUNKSIZE=1` makes each game an independently scheduled pool task. Root exploration and train/eval rollout probabilities are editable constants near the top of the train file, while `--search-deadline` remains a CLI option.

All MCS and PPO entry points use explicit initialization. `--init-mode checkpoint` requires `--init-checkpoint` to exist and fails otherwise. `--init-mode random` creates fresh random model parameters on every new run and ignores `--init-checkpoint`; it does not create a shared random init checkpoint. MCS still saves the active per-agent population files because those are training state and resume outputs.

```bash
python3 train_mcs_selfplay_memory.py \
  --init-mode random \
  --population-dir artifacts/b13902102/population_memory_random \
  --num-workers 1
```

`train_ppo_selfplay.py` is retained as the original PPO-lite experiment. `train_ppo_onpolicy.py` collects a fresh rollout batch with a frozen policy, performs several PPO epochs on only that batch, then discards it. Each game assigns one random learner seat; the other seats use frozen current-policy copies, historical policies, or configured baseline agents, and only learner decisions enter the PPO batch. Use `--profile train` for broader self-play updates or `--profile finetune` for smaller updates from a strong BC/MCS checkpoint. `--num-workers N` uses persistent spawned workers with shared models, so rollout iterations do not write temporary model files.

Both PPO entry points record `initialization=checkpoint` or `initialization=random` in their output checkpoint metadata.

PPO keeps up to 10 in-memory historical snapshots. The profile or CLI controls snapshot interval, minimum/maximum iteration age, and recent-history preference. Fixed opponents are configured separately through `BASELINE_OPPONENTS` and `BASELINE_OPPONENT_ARGS` near the top of `train_ppo_onpolicy.py`, using the same `(module, class, weight)` style as BC teachers. Use `--disable-history` or `--disable-baselines` to disable either source.

```bash
python3 train_ppo_onpolicy.py \
  --profile finetune \
  --init-mode checkpoint \
  --init-checkpoint artifacts/b13902102/population/archive/best.pth \
  --output-checkpoint artifacts/b13902102/ppo_onpolicy_finetune.pth \
  --num-workers 4
```

`train_bc.py` defaults to fast teacher collection for slow local search teachers and checkpoints the dataset every 1000 completed games. Use `--save-every-games N` to change the interval. To resume or extend a dataset, pass `--existing-dataset old.pt`; new decisions are appended and written to `--dataset`. The two paths may be the same.

```bash
python3 train_bc.py \
  --num-games 1000 \
  --existing-dataset artifacts/b13902102/bc_dataset.pt \
  --dataset artifacts/b13902102/bc_dataset.pt \
  --save-every-games 50
```

Add `--full-teachers` when you want slower full-budget teacher data.

Outcome-weighted BC is available as a separate pipeline with different default files:

```bash
python3 train_bc_weighted.py \
  --num-games 2000 \
  --dataset artifacts/b13902102/bc_weighted_dataset.pt \
  --checkpoint artifacts/b13902102/bc_weighted_policy.pth
```

It stores per-decision outcome, teacher, rank, and final-score metadata. Rank and score favor successful trajectories, teacher quality applies a smaller correction, and each game's unclipped relative player weights are normalized to mean `1.0`. The weighted pipeline is self-contained: `WEIGHTED_TEACHERS` defines each teacher's module, class, collection sampling weight, and loss quality weight; its own `FAST_TEACHER_ARGS` and `--full-teachers` control collection budgets. Weighted collection/training defaults live in `WEIGHTED_BC_CONFIG`; `--batch-size`, `--learning-rate`, `--entropy-coef`, `--label-smoothing`, and weighting arguments can override them.

Model-based player entries:

```json
["src.players.b13902102.policy_model_player", "PolicyModelPlayer", {}, "policy"]
["src.players.b13902102.mcts_model_player", "MCTSModelPlayer", {"sims": 48, "deadline_s": 0.72}, "mcts-policy"]
```

For `MCTSModelPlayer`, `sims` is the maximum number of root simulations and `deadline_s` is a per-action soft wall-clock limit. The search stops at whichever limit is reached first. Keep `deadline_s` below the engine's hard timeout to leave time for feature extraction, model inference, and Python overhead.
