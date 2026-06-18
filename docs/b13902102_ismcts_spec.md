# Belief-weighted ISMCTS Agent

## Purpose

`BeliefISMCTSPlayer` is a model-free search agent for the simultaneous,
imperfect-information decisions in 6 Nimmt. It searches only the root
player's decisions. The three opponents choose cards simultaneously from
sampled hidden hands, so no simulated opponent can observe another player's
card from the same round.

The implementation is split into small modules:

- `ismcts_player.py`: grader-facing agent, configuration, fallback, and
  cross-round subtree reuse.
- `ismcts_search.py`: information-set nodes, PUCT, paired root warm-up,
  observation progressive widening, rollouts, and final robust selection.
- `ismcts_belief.py`: hidden-hand proposals, history likelihood weighting,
  systematic resampling, and opponent-profile posterior estimation.
- `ismcts_policy.py`: fast model-free priors and stochastic opponent/rollout
  policies.
- `ismcts_simulator.py`: compact exact rule simulator using only row tail,
  length, and bullhead total.

## Information sets and transitions

An information-set node stores statistics only for the root player's legal
cards. A simulation performs the following transition:

1. Sample one complete assignment of remaining opponent cards from the belief
   pool.
2. Select the root player's card with PUCT.
3. Sample all three opponent cards from the same pre-action public state.
4. Resolve all four cards in ascending order with the exact engine rules.
5. Enter a child information set identified by public board summaries,
   scores, round, and the root player's remaining hand.

Opponent cards are stochastic transition outcomes rather than sequential tree
actions. This prevents information leakage in a simultaneous round and keeps
the tree small enough for the one-second action limit.

## Hidden-hand belief

Cards in the current hand, current board, previous boards, and previous
actions are removed from the unknown pool. Each proposal assigns the required
number of remaining cards to every opponent; unused cards represent undealt
cards.

For every proposed hand, past public rounds are replayed as observations. The
likelihood of each opponent's actual action is evaluated under five profiles:

- `safe`: strongly avoids immediate penalties and dangerous fifth cards.
- `flexible`: retains future playable cards.
- `dump`: safely removes large or high-bullhead cards.
- `sacrifice`: permits cheap low-card row takes.
- `random`: protects against a misspecified opponent model.

Profile likelihoods are marginalized to score each proposed hidden hand.
Likelihood tempering defaults to `0.55`, and `15%` uniform belief mass is
retained to prevent particle collapse. The same calculation produces an
independent posterior mixture of opponent profiles for each seat.

## Search enhancements

### Heuristic PUCT

The root-player prior combines safe, flexible, and dumping policies.
`c_puct=2.25` is intentionally exploratory because there are at most ten
legal actions and the action deadline is the primary search bound.

### Paired root warm-up

Before normal tree selection, every root card receives three full rollouts.
All cards in one pass use the same determinization and random seed. Opponent
choices are therefore paired across candidate cards, reducing variance in
their comparison.

### Observation progressive widening

One action can produce many different public boards. The number of explicit
children grows as:

```text
2.4 * action_visits ^ 0.52
```

and is capped at 18. Additional observations are mapped to the closest
existing public-state cluster. Exact observation keys remain available for
cross-round subtree reuse.

### Heavy stochastic rollouts

Rollouts continue to the end of the game. The root player uses a mixed
heuristic policy with `18%` random exploration. Opponents use their inferred
profile mixture with `22%` random exploration. This avoids the brittle bias
of both fully random and fully greedy simulations.

### Utility and final selection

Terminal utility is dominated by pairwise rank outcomes and includes smaller
score-margin and absolute-score terms. Final selection uses mean utility,
standard error, catastrophic-loss rate, and a small visit-confidence bonus.
Risk penalties increase while leading and decrease while trailing.

## Default exploration budget

The defaults deliberately request more work than normally fits in one move:

```text
deadline_s          = 0.9
max_iterations      = 20000
c_puct              = 2.25
root_warmup_passes  = 3
particle_count      = 128
proposal_count      = 192
tree_horizon        = 5
```

The absolute `0.9s` deadline is authoritative; `max_iterations` is only a
safety ceiling. Belief construction is separately capped so that tree search
retains most of the action budget.

## Config entry

```json
[
    "src.players.b13902102.ismcts_player",
    "BeliefISMCTSPlayer",
    {
        "deadline_s": 0.9,
        "max_iterations": 20000,
        "c_puct": 2.25,
        "root_warmup_passes": 3,
        "particle_count": 128,
        "proposal_count": 192,
        "tree_horizon": 5
    },
    "belief-ismcts"
]
```

`configs/game/ismcts.json` provides a direct comparison with both legacy best
players and Baseline10. `configs/tournament/ismcts.json` provides a larger
cycle-duplicated comparison against selected public baselines and the existing
model-free agents.

## Practical tuning order

After collecting enough duplicated tournament games, tune parameters in this
order:

1. `deadline_s` only if the environment reports timeouts.
2. `opponent_epsilon` and `rollout_epsilon` for opponent-model robustness.
3. `c_puct` and `root_warmup_passes` for exploration allocation.
4. `belief_temperature` and `belief_uniform_mix` for hidden-hand inference.
5. `risk_penalty` and `catastrophe_penalty` for rank versus score behavior.

The current values are search-oriented initial defaults, not empirically
fitted tournament parameters.
