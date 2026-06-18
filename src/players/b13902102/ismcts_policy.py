import math

from .rules import BULLHEAD_TABLE, BOARD_WIDTH


PROFILE_NAMES = ("safe", "flexible", "dump", "sacrifice", "random")
DEFAULT_PROFILE_PRIOR = (0.34, 0.28, 0.14, 0.14, 0.10)


def _placement_profile(state, card):
    best_idx = -1
    best_tail = -1
    for idx, row in enumerate(state.rows):
        if row.tail < card and row.tail > best_tail:
            best_idx = idx
            best_tail = row.tail

    if best_idx < 0:
        best_idx = min(
            range(len(state.rows)),
            key=lambda idx: (
                state.rows[idx].penalty,
                state.rows[idx].count,
                idx,
            ),
        )
        row = state.rows[best_idx]
        return row.penalty, best_idx, 1, BULLHEAD_TABLE[card], True

    row = state.rows[best_idx]
    if row.count >= BOARD_WIDTH:
        return row.penalty, best_idx, 1, BULLHEAD_TABLE[card], False
    return 0, best_idx, row.count + 1, row.penalty + BULLHEAD_TABLE[card], False


def _future_flexibility(card, hand, state):
    remaining = [other for other in hand if other != card]
    if not remaining:
        return 0.0
    penalty, target, new_count, _new_penalty, _low = _placement_profile(state, card)
    tails = [row.tail for row in state.rows]
    tails[target] = card
    counts = [row.count for row in state.rows]
    counts[target] = new_count
    safe = 0.0
    for other in remaining:
        candidates = [(tail, idx) for idx, tail in enumerate(tails) if tail < other]
        if not candidates:
            continue
        _, idx = max(candidates)
        safe += 1.0 if counts[idx] < BOARD_WIDTH else 0.12
    return safe / len(remaining) - 0.15 * penalty


def profile_action_scores(hand, state, player_idx, profile):
    if profile == "random":
        return [0.0] * len(hand)

    scores = []
    mean_card = sum(hand) / len(hand)
    for card in hand:
        penalty, target, new_count, new_penalty, low_take = _placement_profile(state, card)
        pressure = 0.0
        if new_count >= BOARD_WIDTH:
            pressure = 2.6 + 0.10 * new_penalty
        elif new_count == BOARD_WIDTH - 1:
            pressure = 1.0 + 0.05 * new_penalty
        flexibility = _future_flexibility(card, hand, state)
        high = card / 104.0
        bull = BULLHEAD_TABLE[card]

        if profile == "safe":
            value = -4.8 * penalty - 1.15 * pressure + 0.30 * flexibility
            value -= 0.08 * bull
        elif profile == "flexible":
            value = -3.8 * penalty - 0.75 * pressure + 1.30 * flexibility
            value -= 0.002 * abs(card - mean_card)
        elif profile == "dump":
            value = -4.0 * penalty - 0.75 * pressure + 1.15 * high
            value += 0.10 * bull + 0.20 * flexibility
        else:
            cheap_take = low_take and penalty <= 3
            sacrifice_bonus = 2.2 if cheap_take else 0.0
            value = -3.5 * penalty - 0.80 * pressure + sacrifice_bonus
            value += 0.30 * high + 0.15 * bull + 0.20 * flexibility
        scores.append(value)
    return scores


def _softmax_probabilities(values, temperature):
    if not values:
        return []
    temperature = max(0.05, float(temperature))
    maximum = max(values)
    weights = [math.exp(max(-40.0, min(0.0, (value - maximum) / temperature))) for value in values]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [weight / total for weight in weights]


def policy_probability(card, hand, state, player_idx, profile, temperature=0.72, epsilon=0.08):
    if card not in hand or not hand:
        return 1.0e-12
    values = profile_action_scores(hand, state, player_idx, profile)
    probabilities = _softmax_probabilities(values, temperature)
    index = hand.index(card)
    return epsilon / len(hand) + (1.0 - epsilon) * probabilities[index]


def sample_profile(profile_weights, rng):
    pick = rng.random()
    total = 0.0
    for name, weight in zip(PROFILE_NAMES, profile_weights):
        total += weight
        if pick <= total:
            return name
    return PROFILE_NAMES[-1]


def sample_opponent_action(
    hand,
    state,
    player_idx,
    profile_weights,
    rng,
    temperature=0.78,
    epsilon=0.22,
):
    if len(hand) <= 1:
        return hand[0]
    if rng.random() < epsilon:
        return rng.choice(hand)
    profile = sample_profile(profile_weights, rng)
    values = profile_action_scores(hand, state, player_idx, profile)
    probabilities = _softmax_probabilities(values, temperature)
    pick = rng.random()
    total = 0.0
    for card, probability in zip(hand, probabilities):
        total += probability
        if pick <= total:
            return card
    return hand[-1]


def own_action_prior(hand, state, player_idx, temperature=0.65):
    safe = profile_action_scores(hand, state, player_idx, "safe")
    flexible = profile_action_scores(hand, state, player_idx, "flexible")
    dump = profile_action_scores(hand, state, player_idx, "dump")
    values = [
        0.48 * safe_value + 0.38 * flex_value + 0.14 * dump_value
        for safe_value, flex_value, dump_value in zip(safe, flexible, dump)
    ]
    probabilities = _softmax_probabilities(values, temperature)
    return {card: probability for card, probability in zip(hand, probabilities)}


def sample_own_rollout_action(hand, state, player_idx, rng, temperature=0.62, epsilon=0.18):
    if len(hand) <= 1:
        return hand[0]
    if rng.random() < epsilon:
        return rng.choice(hand)
    prior = own_action_prior(hand, state, player_idx, temperature)
    pick = rng.random()
    total = 0.0
    for card in hand:
        total += prior[card]
        if pick <= total:
            return card
    return hand[-1]
