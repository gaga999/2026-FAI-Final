import math
import random

import numpy as np
import torch

from .heuristic import heuristic_select
from .rl_features import action_index, extract_policy_features, index_to_card
from .rl_model import masked_policy, select_card_from_logits
from .rl_train_env import hidden_state_from_history


class MCSConfig:
    def __init__(
        self,
        sims=48,
        c_puct=1.4,
        prior_temperature=1.0,
        rollout_temperature=1.0,
        root_temperature=1.0,
        heuristic_mix=0.25,
        random_mix=0.10,
    ):
        self.sims = int(sims)
        self.c_puct = float(c_puct)
        self.prior_temperature = float(prior_temperature)
        self.rollout_temperature = float(rollout_temperature)
        self.root_temperature = float(root_temperature)
        self.heuristic_mix = float(heuristic_mix)
        self.random_mix = float(random_mix)


def root_mcs_search(model, hand, history, player_idx, config=None, rng=None):
    config = MCSConfig() if config is None else config
    rng = random.Random() if rng is None else rng
    cards = sorted(int(card) for card in hand)
    if not cards:
        pi = np.zeros(104, dtype=np.float32)
        return 1, pi, {}

    features, legal_mask_np = extract_policy_features(cards, history, player_idx)
    with torch.no_grad():
        x = torch.from_numpy(features).float().unsqueeze(0)
        legal = torch.from_numpy(legal_mask_np).bool().unsqueeze(0)
        logits = model(x)
        prior = masked_policy(logits, legal, temperature=config.prior_temperature).squeeze(0).cpu().numpy()

    visits = {card: 0 for card in cards}
    value_sum = {card: 0.0 for card in cards}
    priors = {card: max(float(prior[action_index(card)]), 1.0e-8) for card in cards}

    for _ in range(config.sims):
        root_card = _select_puct_card(cards, visits, value_sum, priors, config.c_puct)
        state = hidden_state_from_history(cards, history, player_idx, rng)
        value = _simulate_with_forced_root(model, state, player_idx, root_card, config, rng)
        visits[root_card] += 1
        value_sum[root_card] += value

    pi = np.zeros(104, dtype=np.float32)
    total_visits = max(1, sum(visits.values()))
    for card, count in visits.items():
        pi[action_index(card)] = count / total_visits

    action = _sample_from_visit_distribution(pi, cards, config.root_temperature, rng)
    stats = {"visits": visits, "value_sum": value_sum, "priors": priors}
    return action, pi, stats


def _select_puct_card(cards, visits, value_sum, priors, c_puct):
    total = max(1, sum(visits.values()))
    best_card = cards[0]
    best_value = float("-inf")
    for card in cards:
        q = value_sum[card] / visits[card] if visits[card] > 0 else 0.0
        u = c_puct * priors[card] * math.sqrt(total) / (1 + visits[card])
        value = q + u
        if value > best_value or (value == best_value and card < best_card):
            best_value = value
            best_card = card
    return best_card


def _simulate_with_forced_root(model, state, player_idx, root_card, config, rng):
    if root_card not in state.hands[player_idx]:
        root_card = min(state.hands[player_idx])
    first = True
    while not state.is_terminal():
        history = state.history_for_player()
        actions = []
        for idx, hand in enumerate(state.hands):
            if not hand:
                continue
            if idx == player_idx and first:
                card = root_card
            else:
                card = _rollout_action(model, hand, history, idx, config, rng)
            actions.append((idx, card))
        state.step(actions)
        first = False
    return state.utility(player_idx)


def _rollout_action(model, hand, history, player_idx, config, rng):
    if len(hand) == 1:
        return hand[0]
    r = rng.random()
    if r < config.random_mix:
        return rng.choice(hand)
    if r < config.random_mix + config.heuristic_mix:
        return heuristic_select(hand, history, player_idx)

    features, legal_mask_np = extract_policy_features(hand, history, player_idx)
    with torch.no_grad():
        x = torch.from_numpy(features).float().unsqueeze(0)
        legal = torch.from_numpy(legal_mask_np).bool().unsqueeze(0)
        logits = model(x).squeeze(0)
        card = select_card_from_logits(logits, legal.squeeze(0), temperature=config.rollout_temperature, sample=True)
    return card if card in hand else min(hand)


def _sample_from_visit_distribution(pi, cards, temperature, rng):
    if temperature <= 1.0e-6:
        return max(cards, key=lambda card: (pi[action_index(card)], -card))
    weights = []
    for card in cards:
        weights.append(float(pi[action_index(card)]) ** (1.0 / max(temperature, 1.0e-6)))
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(cards)
    pick = rng.random() * total
    acc = 0.0
    for card, weight in zip(cards, weights):
        acc += weight
        if pick <= acc:
            return card
    return cards[-1]
