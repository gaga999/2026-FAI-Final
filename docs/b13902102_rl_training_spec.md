# b13902102 RL / MCS Training Spec

This document describes the model-based training implementation for the 6 Nimmt project.

The design follows the final agreed scheme:

1. Weighted behavior cloning warmup from black-box baselines and local agents.
2. Alpha0.5-style population self-play with root MCS / PUCT search targets.
3. Optional PPO-lite fine-tuning, kept outside the main replacement loop.

The final submitted model agents live under `src/players/b13902102/` and load checkpoints relative to their own file directory with `Path(__file__).resolve().parent`.

## Model I/O

The policy model is a fixed-size MLP actor:

```text
input: fixed feature vector, FEATURE_DIM = 800
output: 104 logits, index i means card i + 1
```

Illegal cards are masked at action time, so the model can always output 104 logits.

The model has no value head in the submitted architecture. Stage 2 uses MCS/PUCT to produce improved policy targets instead of relying on a learned value function.

## Feature Scheme

Implemented in:

```text
src/players/b13902102/rl_features.py
```

Main functions:

```python
extract_policy_features(hand, history, player_idx) -> tuple[np.ndarray, np.ndarray]
legal_mask_from_hand(hand) -> np.ndarray
action_index(card) -> int
index_to_card(index) -> int
```

Feature vector layout:

```text
hand_mask: 104
played_mask: 104
board_mask: 104
board_slots: 4 rows * 5 slots * [card_norm, occupied] = 40
row_summary: 4 rows * [tail, length, row_score, slots_left, danger] = 20
sorted_row_summary: 4 rows sorted by tail * [tail, length, row_score] = 12
context: 20
dynamic_action_planes: 4 * 104 = 416
total: 800
```

Dynamic action planes:

```text
immediate_penalty_if_played
target_row_length
gap_to_target_tail
approx_take_probability
```

Constant per-card planes such as card number or bullhead count are intentionally not included. Since the output head has a fixed logit per card, static card identity preferences can be learned by output-specific weights and biases. The 104-wide planes only encode state-dependent action information.

## Model Module

Implemented in:

```text
src/players/b13902102/rl_model.py
```

Important objects:

```python
class PolicyMLP
mask_logits(logits, legal_mask)
masked_policy(logits, legal_mask, temperature)
select_card_from_logits(logits, legal_mask, temperature, sample)
supervised_ce_loss(...)
policy_loss(...)
load_policy_model(checkpoint_path)
save_policy_checkpoint(path, model, optimizer=None, **metadata)
```

`PolicyMLP` uses:

```text
800 -> 512 -> 256 -> 256 -> 104
LayerNorm + SiLU after each hidden layer
```

## Lightweight Training Env

Implemented in:

```text
src/players/b13902102/rl_train_env.py
```

Important objects:

```python
class LightGameState
deal_light_game(seed=None)
hidden_state_from_history(hand, history, player_idx, rng)
```

`LightGameState` mirrors the official engine rules but is designed for training and rollout:

```python
state.clone()
state.history_for_player()
state.step(actions)
state.ranks()
state.utility(player_idx)
```

The utility is rank-first:

```text
utility = (2.5 - rank) - 0.03 * score - 0.02 * max(0, score - mean_opponent_score)
```

This aligns training with tournament average rank while using score as a tie/stability signal.

## Root MCS / PUCT

Implemented in:

```text
src/players/b13902102/rl_mcs.py
```

Important objects:

```python
class MCSConfig
root_mcs_search(model, hand, history, player_idx, config, rng)
```

Search procedure:

1. Extract features and get model logits.
2. Mask illegal cards and compute prior probabilities with `prior_temperature`.
3. For each simulation, select a root action with:

```text
Q(a) + c_puct * P(a) * sqrt(total_visits) / (1 + N(a))
```

4. Sample hidden opponent hands from unseen cards.
5. Force the root action.
6. Roll out the rest of the game using a policy / heuristic / random mixture.
7. Update root visit counts and Q values.
8. Return the selected action and a 104-dim normalized visit distribution. `root_temperature` controls sampling from visit counts; set it to `0` for deterministic evaluation.

Stage 2 trains the policy to imitate this visit distribution.

## Stage 1: Behavior Cloning

Root script:

```text
train_bc.py
```

Main functions:

```python
collect_bc_dataset(
    num_games,
    output_path,
    seed=0,
    existing_dataset_path=None,
    save_every_games=100,
)
train_bc(dataset_path, checkpoint_path, epochs=5, ...)
```

Data collection uses a `RecordingPlayer` wrapper:

1. Sample 4 teacher agents with weights.
2. Run one official `Engine` game.
3. Before each teacher decision, extract features and legal mask.
4. After teacher action, record `(features, legal_mask, action_idx)`.

Teacher sampling weights are intentionally uneven:

```text
high: best_player1, best_player2, adaptive_portfolio
medium: public baselines 6-10
low: pure heuristic and public baselines 1-5
```

The black-box baseline `.so` players are treated like normal imported Python classes; the wrapper only observes their input/output.

By default, `train_bc.py` uses a fast teacher mode for local search teachers (`best_player1`, `best_player2`, adaptive portfolio, rollout, beam) so a default BC run starts producing data quickly. Use `--full-teachers` to disable the reduced deadline/rollout budgets and collect slower but stronger teacher behavior.

Collection durability:

1. `--save-every-games N` writes a dataset checkpoint after every `N` completed games.
2. Each checkpoint atomically replaces the output through a temporary file.
3. `--existing-dataset PATH` loads an earlier dataset and appends newly collected records.
4. `--existing-dataset` and `--dataset` may point to the same file for in-place continuation.
5. An interruption loses at most the unfinished game plus records collected since the last checkpoint.

Training loss:

```text
cross entropy on masked logits
+ label smoothing over legal actions
- entropy bonus
```

Default output:

```text
artifacts/b13902102/bc_dataset.pt
artifacts/b13902102/bc_policy.pth
```

### Weighted BC Variant

`train_bc_weighted.py` is an independent entry point and uses:

```text
artifacts/b13902102/bc_weighted_dataset.pt
artifacts/b13902102/bc_weighted_policy.pth
```

It keeps all legal teacher decisions but assigns every player's full-game trajectory a weight. The default rank weights are `1.70, 1.15, 0.75, 0.40`; an exponential final-score factor and a teacher-quality factor are then applied. These factors are not clipped. The four player weights are normalized to mean `1.0` inside each game, so their relative weighting is preserved without allowing one game to contribute more total loss merely because its raw factors are large.

The dataset additionally stores:

```text
weights, outcome_weights, teacher_weights, ranks, final_scores, teacher_ids
```

Training minimizes normalized weighted cross entropy plus the existing entropy regularizer. `--teacher-weight-strength 0` disables teacher loss weighting while retaining outcome weighting.

The weighted script is independent from `train_bc.py`. Its `WEIGHTED_TEACHERS` entries use:

```text
(module_path, class_name, collection_sampling_weight, loss_quality_weight)
```

This keeps both teacher-specific weighting controls in one array without importing BC configuration. Its local `FAST_TEACHER_ARGS` reduces BestPlayer1/2 rollout budgets and deadlines and reduces the other local search-agent deadlines/budgets; `--full-teachers` disables those overrides.

Weighted collection/training defaults are grouped in `WEIGHTED_BC_CONFIG`. CLI options expose the dataset checkpoint interval, epochs, batch size, learning rate, entropy coefficient, label smoothing, score coefficient, and teacher-weight strength without requiring code edits.

## Stage 2: Population MCS Self-Play

Root script:

```text
train_mcs_selfplay.py
```

Core module:

```text
src/players/b13902102/rl_population.py
```

Important objects:

```python
class AgentSlot
class PopulationTrainer
PopulationTrainer.initialize_from_checkpoint(...)
PopulationTrainer.initialize_random(...)
PopulationTrainer.train_selfplay_games(...)
PopulationTrainer.evaluate_frozen(...)
PopulationTrainer.replace_bottom(...)
PopulationTrainer.save_archive(...)
compute_bradley_terry_elo(...)
```

Population:

```text
8 active agents
each has its own checkpoint, optimizer state, replay buffer, temperature, c_puct, entropy coefficient, EMA rating
```

Initialization:

```text
--init-mode checkpoint:
    the required checkpoint is cloned into 8 agents
--init-mode random:
    each of the 8 agents receives a separately constructed fresh PolicyMLP
    no shared random init checkpoint is created
different temperatures, c_puct, and entropy settings create additional diversity
```

A missing checkpoint is an error in checkpoint mode. Random initialization is never inferred from a missing path. It must be selected explicitly, and every new random-mode invocation reconstructs the models from new initial parameters. The active `agent_N.pth` files written under the population directory are normal training/resume state, not a reusable random init file.

Training self-play:

```text
sample 4 active agents
play a light-env game
each decision runs root MCS / PUCT
store (features, legal_mask, pi_mcs)
update each agent with policy_loss(logits, pi_mcs)
```

Matchmaking:

```text
70% uniform random matchup
20% top-challenge matchup
10% parent-lineage diversity matchup
```

Frozen evaluation:

```text
no learning
fixed checkpoints
random partitions of active population
cycle hand duplication per table
rank-based scoring
```

Per-game points:

```text
points = 4.5 - rank
rank 1 -> 3.5
rank 2 -> 2.5
rank 3 -> 1.5
rank 4 -> 0.5
```

Replacement uses EMA points/rank, not live training-game results:

```text
ema_points = 0.75 * old + 0.25 * current_eval_points
ema_rank   = 0.75 * old + 0.25 * current_eval_rank
```

Replacement mechanism:

```text
protect top 2 elites
replace bottom 1 by default after full eval
70% clone top 1
30% clone one of top 2-3
reset victim replay
reset optimizer via checkpoint clone
mutate temperature, c_puct, entropy coefficient
give one eval grace period
archive top checkpoints
```

Elo:

`compute_bradley_terry_elo()` computes diagnostic Elo from pairwise rank outcomes during frozen evaluation. Elo is recorded for analysis, but replacement is based on EMA points/rank because that is closer to the official tournament objective.

### Memory-First Parallel MCS Variant

`train_mcs_selfplay_memory.py` and `rl_population_memory.py` retain the same eight-agent tournament scheme while changing collection and update execution:

```text
freeze the current population snapshot
collect one generation of games, optionally in spawned CPU workers
append targets to a 6000-row recent replay buffer per agent
perform 96 in-memory optimizer steps per agent
run frozen evaluation
save only at game intervals, replacement/archive events, and final shutdown
```

Workers never update model parameters. Persistent spawned workers read shared frozen CPU model copies, while the main process owns all live models and optimizers. The main process synchronizes the shared copies only between complete collection/evaluation batches. `torch.set_num_threads(1)` is used inside workers to avoid CPU oversubscription.

Pool scheduling and search probabilities are configured near the top of the root train scripts rather than through CLI flags:

```text
POOL_CHUNKSIZE = 1
ROOT_EXPLORATION_MIX
TRAIN_HEURISTIC_MIX / TRAIN_RANDOM_MIX
EVAL_HEURISTIC_MIX / EVAL_RANDOM_MIX
```

With `POOL_CHUNKSIZE=1`, every training game is independently placed on the process-pool work queue, so a free worker immediately takes the next game. Model rollout probability is the remainder `1 - heuristic_mix - random_mix`. Both MCS entry points expose only the per-decision time limit as `--search-deadline`; setting it to `0` leaves the simulation count as the sole search limit.

Default decision diversity comes from:

```text
agent-specific root visit temperatures: 1.25, 1.10, 0.95, 0.80
3% uniform legal-action mixture during training action selection
stochastic policy/random rollout mixture inside MCS
agent-specific entropy regularization from about 0.0018 to 0.0045
```

The replay buffer is deliberately smaller than the original 20000 rows so targets stay closer to the current population. With batch size 256, the default 96 updates per agent processes approximately the same number of training samples per generation as the original 192 updates with batch size 128. The default MCS deadline remains 0.5 seconds. Replay is not written to disk; resuming restores models, optimizers, rankings, RNG state, and population metadata with an empty replay buffer. Active population checkpoints use temporary-file replacement, while archive checkpoints omit optimizer state to reduce file size.

Example:

```bash
python3 train_mcs_selfplay_memory.py \
  --init-mode checkpoint \
  --init-checkpoint artifacts/b13902102/bc_policy.pth \
  --population-dir artifacts/b13902102/population_memory \
  --num-workers 4 \
  --save-every-games 512
```

## Stage 3: Optional PPO-Lite

Root script:

```text
train_ppo_selfplay.py
```

This is intentionally not the main training path. It provides actor-only PPO-style fine-tuning from a BC/MCS checkpoint:

```text
sample self-play episodes
store old logprob, action, terminal utility
normalize terminal returns as advantages
clipped policy objective + entropy
```

Use this only after MCS population training produces a stable checkpoint.

### On-Policy PPO Variant

`train_ppo_onpolicy.py` keeps the original PPO-lite script unchanged while using a strict rollout/update cycle:

```text
copy the learner into a shared frozen rollout model
collect a fresh batch from one learner seat per game
run several PPO epochs only on that batch
discard the batch before the next rollout iteration
```

No rollout checkpoint is written to disk. Persistent spawned workers read one shared current model and a ring of 10 shared historical models. A snapshot is added at the configured iteration interval; historical sampling applies minimum/maximum age bounds and a configurable preference for recent snapshots.

Fixed baseline opponents are separate from historical policies:

```text
BASELINE_OPPONENTS = [(module_path, class_name, sampling_weight), ...]
BASELINE_OPPONENT_ARGS = {(module_path, class_name): constructor_kwargs, ...}
```

This matches the BC teacher configuration style. Current-policy, historical-policy, and baseline games have separate probabilities. Only actions sampled by the frozen learner policy are stored in the PPO batch.

Both PPO entry points use the same explicit `--init-mode checkpoint|random` contract as MCS. Checkpoint mode requires the file to exist. Random mode creates a fresh `PolicyMLP` on every invocation and records `initialization=random`; no random init checkpoint is written. Missing MCS resume state remains an error because silently restarting would discard tournament state.

## Submitted / Evaluation Agents

Implemented in:

```text
src/players/b13902102/policy_model_player.py
src/players/b13902102/mcts_model_player.py
```

`PolicyModelPlayer`:

```text
load checkpoint from same folder
extract features
mask illegal cards
argmax or temperature sample
```

Default checkpoint:

```text
src/players/b13902102/policy_model.pth
```

`MCTSModelPlayer`:

```text
load checkpoint from same folder
use model logits as PUCT prior
run root MCS
return selected legal card
```

Default checkpoint:

```text
src/players/b13902102/mcts_model.pth
```

Both agents load with:

```python
HERE = Path(__file__).resolve().parent
ckpt = HERE / checkpoint
```

This makes checkpoint loading independent of the grader's working directory.

## Suggested Pipeline

Behavior cloning warmup:

```bash
python3 train_bc.py \
  --num-games 2000 \
  --dataset artifacts/b13902102/bc_dataset.pt \
  --save-every-games 100 \
  --checkpoint artifacts/b13902102/bc_policy.pth
```

Continue an existing collection:

```bash
python3 train_bc.py \
  --num-games 1000 \
  --existing-dataset artifacts/b13902102/bc_dataset.pt \
  --dataset artifacts/b13902102/bc_dataset.pt \
  --save-every-games 50 \
  --checkpoint artifacts/b13902102/bc_policy.pth
```

The script prints phase boundaries and uses `tqdm` for game and epoch progress.

Weighted BC:

```bash
python3 train_bc_weighted.py \
  --num-games 2000 \
  --dataset artifacts/b13902102/bc_weighted_dataset.pt \
  --checkpoint artifacts/b13902102/bc_weighted_policy.pth
```

MCS population self-play:

```bash
python3 train_mcs_selfplay.py \
  --init-mode checkpoint \
  --init-checkpoint artifacts/b13902102/bc_policy.pth \
  --population-dir artifacts/b13902102/population \
  --generations 40
```

Optional PPO-lite:

```bash
python3 train_ppo_selfplay.py \
  --init-mode checkpoint \
  --init-checkpoint artifacts/b13902102/population/archive/best.pth \
  --output-checkpoint artifacts/b13902102/ppo_policy.pth
```

After choosing final checkpoints, copy them into:

```text
src/players/b13902102/policy_model.pth
src/players/b13902102/mcts_model.pth
```

Then configure players as:

```json
["src.players.b13902102.policy_model_player", "PolicyModelPlayer", {}, "policy"]
["src.players.b13902102.mcts_model_player", "MCTSModelPlayer", {"sims": 48, "deadline_s": 0.72}, "mcts-policy"]
```

`MCTSModelPlayer` time-related controls:

- `sims`: hard upper bound on completed root simulations. One simulation may still be expensive because it rolls out the remaining game.
- `deadline_s`: soft wall-clock budget for one `action()` call. It is checked between simulations and inside rollout rounds. The default is `0.72`, leaving a buffer below the engine's one-second hard timeout.
- `heuristic_mix` and `random_mix` in `MCSConfig`: larger values indirectly reduce inference cost because more rollout actions avoid the policy MLP.
- `rollout_temperature`, `root_temperature`, `prior_temperature`, and `c_puct` change search behavior but do not materially bound runtime.
