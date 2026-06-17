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
collect_bc_dataset(num_games, output_path, seed=0)
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
BC checkpoint is cloned into 8 agents
different seed-like metadata, temperatures, c_puct, and entropy settings create diversity
```

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
  --checkpoint artifacts/b13902102/bc_policy.pth
```

MCS population self-play:

```bash
python3 train_mcs_selfplay.py \
  --init-checkpoint artifacts/b13902102/bc_policy.pth \
  --population-dir artifacts/b13902102/population \
  --generations 40
```

Optional PPO-lite:

```bash
python3 train_ppo_selfplay.py \
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
["src.players.b13902102.mcts_model_player", "MCTSModelPlayer", {"sims": 48}, "mcts-policy"]
```
