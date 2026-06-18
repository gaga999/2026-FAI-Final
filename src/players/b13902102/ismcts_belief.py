import bisect
import math
import time
from dataclasses import dataclass

from .ismcts_policy import (
    DEFAULT_PROFILE_PRIOR,
    PROFILE_NAMES,
    policy_probability,
)
from .ismcts_simulator import CompactState, N_PLAYERS
from .rules import (
    extract_unseen_cards,
    get_board_history,
    get_history_matrix,
    get_score_history,
)


@dataclass(slots=True)
class Determinization:
    hands: tuple
    weight: float


class BeliefPool:
    def __init__(self, particles, profile_weights):
        self.particles = particles
        self.profile_weights = profile_weights
        self.cumulative = []
        total = 0.0
        for particle in particles:
            total += particle.weight
            self.cumulative.append(total)
        if total <= 0.0 and particles:
            step = 1.0 / len(particles)
            self.cumulative = [step * (idx + 1) for idx in range(len(particles))]
        elif total > 0.0:
            self.cumulative = [value / total for value in self.cumulative]

    def sample(self, rng):
        if not self.particles:
            raise ValueError("belief pool has no particles")
        index = bisect.bisect_left(self.cumulative, rng.random())
        return self.particles[min(index, len(self.particles) - 1)]


def build_belief_pool(
    hand,
    history,
    player_idx,
    rng,
    particle_count=128,
    proposal_count=192,
    likelihood_temperature=0.55,
    uniform_mix=0.15,
    deadline=None,
):
    cards = sorted(int(card) for card in hand)
    unseen = extract_unseen_cards(cards, history)
    proposal_count = max(int(particle_count), int(proposal_count))
    raw_particles = []
    raw_logs = []
    raw_posteriors = []

    for proposal_idx in range(proposal_count):
        if proposal_idx >= 24 and deadline is not None and time.perf_counter() >= deadline:
            break
        shuffled = list(unseen)
        rng.shuffle(shuffled)
        hands = [[] for _ in range(N_PLAYERS)]
        hands[player_idx] = list(cards)
        cursor = 0
        valid = True
        for idx in range(N_PLAYERS):
            if idx == player_idx:
                continue
            needed = len(cards)
            if cursor + needed > len(shuffled):
                valid = False
                break
            hands[idx] = sorted(shuffled[cursor : cursor + needed])
            cursor += needed
        if not valid:
            continue

        log_weight, posterior = _history_log_likelihood(
            hands,
            history,
            player_idx,
            likelihood_temperature,
        )
        raw_particles.append(tuple(tuple(player_hand) for player_hand in hands))
        raw_logs.append(log_weight)
        raw_posteriors.append(posterior)

    if not raw_particles:
        fallback = [[] for _ in range(N_PLAYERS)]
        fallback[player_idx] = cards
        raw_particles = [tuple(tuple(player_hand) for player_hand in fallback)]
        raw_logs = [0.0]
        raw_posteriors = [[list(DEFAULT_PROFILE_PRIOR) for _ in range(N_PLAYERS)]]

    maximum = max(raw_logs)
    likelihood_weights = [math.exp(max(-60.0, value - maximum)) for value in raw_logs]
    likelihood_total = sum(likelihood_weights)
    if likelihood_total <= 0.0:
        likelihood_weights = [1.0 / len(raw_particles)] * len(raw_particles)
    else:
        likelihood_weights = [value / likelihood_total for value in likelihood_weights]

    uniform = 1.0 / len(raw_particles)
    combined = [
        (1.0 - uniform_mix) * weight + uniform_mix * uniform
        for weight in likelihood_weights
    ]
    selected = _systematic_resample_indices(combined, min(int(particle_count), len(raw_particles)), rng)
    selected_counts = {}
    for index in selected:
        selected_counts[index] = selected_counts.get(index, 0) + 1

    particles = [
        Determinization(raw_particles[index], count / len(selected))
        for index, count in selected_counts.items()
    ]
    profile_weights = _aggregate_profile_posteriors(
        raw_posteriors,
        combined,
        player_idx,
    )
    return BeliefPool(particles, profile_weights)


def _history_log_likelihood(hands, history, root_player_idx, likelihood_temperature):
    matrix = get_history_matrix(history)
    boards = get_board_history(history)
    score_history = get_score_history(history)
    if not matrix or not boards:
        return 0.0, [list(DEFAULT_PROFILE_PRIOR) for _ in range(N_PLAYERS)]

    total_log = 0.0
    posteriors = [list(DEFAULT_PROFILE_PRIOR) for _ in range(N_PLAYERS)]
    rounds = min(len(matrix), len(boards))
    for player_idx in range(N_PLAYERS):
        if player_idx == root_player_idx:
            continue
        original_hand = list(hands[player_idx])
        for round_idx in range(rounds):
            if player_idx < len(matrix[round_idx]):
                original_hand.append(matrix[round_idx][player_idx])
        original_hand = sorted(set(original_hand))

        profile_logs = [math.log(max(weight, 1.0e-12)) for weight in DEFAULT_PROFILE_PRIOR]
        replay_hand = list(original_hand)
        for round_idx in range(rounds):
            if player_idx >= len(matrix[round_idx]):
                continue
            actual = matrix[round_idx][player_idx]
            if actual not in replay_hand:
                continue
            scores = [0] * N_PLAYERS if round_idx == 0 else list(score_history[round_idx - 1])
            while len(scores) < N_PLAYERS:
                scores.append(0)
            state = CompactState.from_public(
                boards[round_idx],
                scores,
                [[] for _ in range(N_PLAYERS)],
                round_idx,
            )
            for profile_idx, profile in enumerate(PROFILE_NAMES):
                probability = policy_probability(
                    actual,
                    replay_hand,
                    state,
                    player_idx,
                    profile,
                )
                profile_logs[profile_idx] += math.log(max(probability, 1.0e-12))
            replay_hand.remove(actual)

        mixture_log = _logsumexp(profile_logs)
        total_log += likelihood_temperature * mixture_log
        normalized = [math.exp(value - mixture_log) for value in profile_logs]
        posteriors[player_idx] = _normalize(normalized)
    return total_log, posteriors


def _aggregate_profile_posteriors(posteriors, particle_weights, root_player_idx):
    aggregate = [[0.0] * len(PROFILE_NAMES) for _ in range(N_PLAYERS)]
    for posterior, particle_weight in zip(posteriors, particle_weights):
        for player_idx in range(N_PLAYERS):
            for profile_idx in range(len(PROFILE_NAMES)):
                aggregate[player_idx][profile_idx] += (
                    particle_weight * posterior[player_idx][profile_idx]
                )
    for player_idx in range(N_PLAYERS):
        if player_idx == root_player_idx:
            aggregate[player_idx] = list(DEFAULT_PROFILE_PRIOR)
        else:
            aggregate[player_idx] = _normalize(aggregate[player_idx])
    return aggregate


def _systematic_resample_indices(weights, count, rng):
    if count <= 0:
        return []
    cumulative = []
    total = 0.0
    for weight in weights:
        total += weight
        cumulative.append(total)
    if total <= 0.0:
        return [rng.randrange(len(weights)) for _ in range(count)]
    cumulative = [value / total for value in cumulative]
    start = rng.random() / count
    indices = []
    cursor = 0
    for sample_idx in range(count):
        target = start + sample_idx / count
        while cursor + 1 < len(cumulative) and cumulative[cursor] < target:
            cursor += 1
        indices.append(cursor)
    return indices


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _normalize(values):
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]
