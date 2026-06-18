import math
import random
import time
from dataclasses import dataclass, field

from .ismcts_belief import build_belief_pool
from .ismcts_policy import (
    own_action_prior,
    sample_opponent_action,
    sample_own_rollout_action,
)
from .ismcts_simulator import (
    CompactState,
    observation_distance,
    public_state_for_hand,
)


@dataclass(slots=True)
class ActionStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    catastrophe_count: int = 0
    children: dict = field(default_factory=dict)

    def update(self, value):
        self.visits += 1
        self.value_sum += value
        self.value_sq_sum += value * value
        if value <= -0.45:
            self.catastrophe_count += 1

    def mean(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def variance(self):
        if self.visits <= 1:
            return 0.0
        mean = self.mean()
        return max(0.0, self.value_sq_sum / self.visits - mean * mean)


@dataclass(slots=True)
class InfoNode:
    visits: int = 0
    actions: dict = field(default_factory=dict)

    def prepare(self, state, player_idx, prior_temperature):
        hand = state.hands[player_idx]
        priors = own_action_prior(hand, state, player_idx, prior_temperature)
        for card in hand:
            if card not in self.actions:
                self.actions[card] = ActionStats(priors.get(card, 1.0 / len(hand)))


class ISMCTSConfig:
    def __init__(
        self,
        deadline_s=0.9,
        max_iterations=20000,
        c_puct=2.25,
        root_warmup_passes=3,
        particle_count=128,
        proposal_count=192,
        belief_temperature=0.55,
        belief_uniform_mix=0.15,
        prior_temperature=0.65,
        opponent_temperature=0.78,
        opponent_epsilon=0.22,
        rollout_temperature=0.62,
        rollout_epsilon=0.18,
        observation_widening_k=2.4,
        observation_widening_alpha=0.52,
        max_observation_children=18,
        tree_horizon=5,
        risk_penalty=0.20,
        catastrophe_penalty=0.12,
    ):
        self.deadline_s = min(0.95, max(0.10, float(deadline_s)))
        self.max_iterations = max(1, int(max_iterations))
        self.c_puct = max(0.05, float(c_puct))
        self.root_warmup_passes = max(0, int(root_warmup_passes))
        self.particle_count = max(8, int(particle_count))
        self.proposal_count = max(self.particle_count, int(proposal_count))
        self.belief_temperature = max(0.0, float(belief_temperature))
        self.belief_uniform_mix = min(1.0, max(0.0, float(belief_uniform_mix)))
        self.prior_temperature = max(0.05, float(prior_temperature))
        self.opponent_temperature = max(0.05, float(opponent_temperature))
        self.opponent_epsilon = min(1.0, max(0.0, float(opponent_epsilon)))
        self.rollout_temperature = max(0.05, float(rollout_temperature))
        self.rollout_epsilon = min(1.0, max(0.0, float(rollout_epsilon)))
        self.observation_widening_k = max(0.1, float(observation_widening_k))
        self.observation_widening_alpha = min(1.0, max(0.05, float(observation_widening_alpha)))
        self.max_observation_children = max(1, int(max_observation_children))
        self.tree_horizon = max(1, int(tree_horizon))
        self.risk_penalty = max(0.0, float(risk_penalty))
        self.catastrophe_penalty = max(0.0, float(catastrophe_penalty))


def search_ismcts(
    hand,
    history,
    player_idx,
    config,
    rng,
    reused_root=None,
):
    start = time.perf_counter()
    deadline = start + config.deadline_s
    cards = sorted(int(card) for card in hand)
    public_state = public_state_for_hand(cards, history, player_idx)
    root = reused_root if reused_root is not None else InfoNode()
    root.prepare(public_state, player_idx, config.prior_temperature)

    belief_deadline = min(
        deadline - min(0.16, 0.22 * config.deadline_s),
        start + min(0.26, 0.30 * config.deadline_s),
    )
    belief = build_belief_pool(
        cards,
        history,
        player_idx,
        rng,
        particle_count=config.particle_count,
        proposal_count=config.proposal_count,
        likelihood_temperature=config.belief_temperature,
        uniform_mix=config.belief_uniform_mix,
        deadline=belief_deadline,
    )

    _paired_root_warmup(root, belief, public_state, player_idx, config, rng, deadline)
    completed = 0
    max_depth = _tree_depth(len(cards), config.tree_horizon)
    while completed < config.max_iterations and time.perf_counter() < deadline:
        particle = belief.sample(rng)
        state = CompactState.from_history(
            [list(player_hand) for player_hand in particle.hands],
            history,
        )
        _tree_simulation(
            state,
            root,
            player_idx,
            belief.profile_weights,
            config,
            rng,
            deadline,
            0,
            max_depth,
        )
        completed += 1

    selected = _select_final_action(root, public_state, player_idx, config)
    diagnostics = {
        "iterations": completed,
        "particles": len(belief.particles),
        "root_visits": root.visits,
        "actions": {
            card: {
                "visits": stats.visits,
                "mean": stats.mean(),
                "variance": stats.variance(),
                "catastrophe_rate": (
                    stats.catastrophe_count / stats.visits if stats.visits else 0.0
                ),
            }
            for card, stats in root.actions.items()
            if card in cards
        },
    }
    return selected, root, diagnostics


def advance_reused_root(previous_root, previous_card, hand, history, player_idx):
    if previous_root is None or previous_card not in previous_root.actions:
        return None
    current = public_state_for_hand(hand, history, player_idx)
    key = current.observation_key(player_idx)
    return previous_root.actions[previous_card].children.get(key)


def _paired_root_warmup(root, belief, public_state, player_idx, config, rng, deadline):
    cards = list(public_state.hands[player_idx])
    for _pass in range(config.root_warmup_passes):
        if time.perf_counter() >= deadline:
            return
        particle = belief.sample(rng)
        shared_seed = rng.randrange(1 << 30)
        for card in cards:
            if time.perf_counter() >= deadline:
                return
            state = CompactState(
                [row.copy() for row in public_state.rows],
                list(public_state.scores),
                [list(player_hand) for player_hand in particle.hands],
                public_state.round_idx,
            )
            paired_rng = random.Random(shared_seed)
            value = _forced_root_rollout(
                state,
                card,
                player_idx,
                belief.profile_weights,
                config,
                paired_rng,
                deadline,
            )
            root.actions[card].update(value)
            root.visits += 1


def _tree_simulation(
    state,
    node,
    player_idx,
    profile_weights,
    config,
    rng,
    deadline,
    depth,
    max_depth,
):
    if state.is_terminal() or time.perf_counter() >= deadline:
        return state.utility(player_idx)

    node.prepare(state, player_idx, config.prior_temperature)
    card = _select_puct(node, state.hands[player_idx], config.c_puct)
    actions = _joint_actions(
        state,
        player_idx,
        card,
        profile_weights,
        config,
        rng,
    )
    state.play_round(actions)

    if state.is_terminal() or time.perf_counter() >= deadline:
        value = state.utility(player_idx)
    elif depth + 1 >= max_depth:
        value = _rollout_to_terminal(
            state,
            player_idx,
            profile_weights,
            config,
            rng,
            deadline,
        )
    else:
        action_stats = node.actions[card]
        child = _observation_child(
            action_stats,
            state.observation_key(player_idx),
            config,
        )
        value = _tree_simulation(
            state,
            child,
            player_idx,
            profile_weights,
            config,
            rng,
            deadline,
            depth + 1,
            max_depth,
        )

    node.actions[card].update(value)
    node.visits += 1
    return value


def _select_puct(node, legal_cards, c_puct):
    total = math.sqrt(node.visits + 1.0)
    best_card = legal_cards[0]
    best_value = float("-inf")
    for card in legal_cards:
        stats = node.actions[card]
        q_value = stats.mean()
        exploration = c_puct * stats.prior * total / (1.0 + stats.visits)
        progressive_bias = 0.08 * stats.prior / (1.0 + stats.visits)
        value = q_value + exploration + progressive_bias
        if value > best_value or (value == best_value and card < best_card):
            best_value = value
            best_card = card
    return best_card


def _observation_child(action_stats, key, config):
    if key in action_stats.children:
        return action_stats.children[key]
    allowed = int(
        config.observation_widening_k
        * max(1, action_stats.visits) ** config.observation_widening_alpha
    )
    allowed = min(config.max_observation_children, max(1, allowed))
    if len(action_stats.children) < allowed:
        child = InfoNode()
        action_stats.children[key] = child
        return child
    nearest_key = min(
        action_stats.children,
        key=lambda existing: observation_distance(existing, key),
    )
    return action_stats.children[nearest_key]


def _joint_actions(state, player_idx, own_card, profile_weights, config, rng):
    actions = [None] * len(state.hands)
    actions[player_idx] = own_card
    for idx, hand in enumerate(state.hands):
        if idx == player_idx or not hand:
            continue
        actions[idx] = sample_opponent_action(
            hand,
            state,
            idx,
            profile_weights[idx],
            rng,
            temperature=config.opponent_temperature,
            epsilon=config.opponent_epsilon,
        )
    return actions


def _forced_root_rollout(
    state,
    root_card,
    player_idx,
    profile_weights,
    config,
    rng,
    deadline,
):
    actions = _joint_actions(
        state,
        player_idx,
        root_card,
        profile_weights,
        config,
        rng,
    )
    state.play_round(actions)
    return _rollout_to_terminal(
        state,
        player_idx,
        profile_weights,
        config,
        rng,
        deadline,
    )


def _rollout_to_terminal(state, player_idx, profile_weights, config, rng, deadline):
    while not state.is_terminal() and time.perf_counter() < deadline:
        own_hand = state.hands[player_idx]
        if not own_hand:
            break
        own_card = sample_own_rollout_action(
            own_hand,
            state,
            player_idx,
            rng,
            temperature=config.rollout_temperature,
            epsilon=config.rollout_epsilon,
        )
        actions = _joint_actions(
            state,
            player_idx,
            own_card,
            profile_weights,
            config,
            rng,
        )
        state.play_round(actions)
    return state.utility(player_idx)


def _tree_depth(hand_size, configured):
    if hand_size <= 5:
        return hand_size
    if hand_size <= 7:
        return min(configured, 5)
    return min(configured, 4)


def _select_final_action(root, public_state, player_idx, config):
    hand = public_state.hands[player_idx]
    scores = public_state.scores
    mine = scores[player_idx]
    opponents = [score for idx, score in enumerate(scores) if idx != player_idx]
    mean_opp = sum(opponents) / len(opponents) if opponents else mine
    if mine + 4 < mean_opp:
        risk_scale = 1.35
    elif mine > mean_opp + 5:
        risk_scale = 0.65
    else:
        risk_scale = 1.0

    best_card = hand[0]
    best_value = float("-inf")
    for card in hand:
        stats = root.actions[card]
        if not stats.visits:
            value = stats.prior - 10.0
        else:
            standard_error = math.sqrt(stats.variance() / stats.visits)
            catastrophe_rate = stats.catastrophe_count / stats.visits
            value = stats.mean()
            value -= risk_scale * config.risk_penalty * standard_error
            value -= risk_scale * config.catastrophe_penalty * catastrophe_rate
            value += 0.008 * math.log1p(stats.visits)
        if value > best_value or (value == best_value and card < best_card):
            best_value = value
            best_card = card
    return best_card
