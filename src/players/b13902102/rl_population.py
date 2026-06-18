import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from .rl_checkpoint import clone_checkpoint, load_checkpoint, save_checkpoint
from .rl_features import action_index, extract_policy_features
from .rl_mcs import MCSConfig, root_mcs_search
from .rl_model import PolicyMLP, policy_loss
from .rl_train_env import deal_light_game


@dataclass
class AgentSlot:
    agent_id: int
    checkpoint: Path
    parent_id: int | None = None
    generation: int = 0
    temperature: float = 1.0
    c_puct: float = 1.4
    entropy_coef: float = 0.003
    ema_points: float = 2.0
    ema_rank: float = 2.5
    elo: float = 1500.0
    grace: int = 0
    bottom_streak: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalRecord:
    avg_rank: float
    avg_points: float
    avg_score: float
    games: int


class PopulationTrainer:
    def __init__(
        self,
        population_dir,
        device="cpu",
        population_size=8,
        seed=0,
        lr=3.0e-4,
        batch_size=128,
        replay_limit=20000,
    ):
        self.population_dir = Path(population_dir)
        self.population_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.population_size = int(population_size)
        self.rng = random.Random(seed)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.replay_limit = int(replay_limit)
        self.slots = []
        self.replay = {}

    def initialize_from_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {checkpoint_path}"
            )
        self.slots = []
        self.replay = {}
        for agent_id in range(self.population_size):
            dst = self.population_dir / f"agent_{agent_id}.pth"
            clone_checkpoint(
                checkpoint_path,
                dst,
                {
                    "agent_id": agent_id,
                    "parent_id": None,
                    "initialization": "checkpoint",
                    "init_checkpoint": str(checkpoint_path),
                    "temperature": self._initial_temperature(agent_id),
                },
            )
            self.slots.append(
                AgentSlot(
                    agent_id=agent_id,
                    checkpoint=dst,
                    parent_id=None,
                    temperature=self._initial_temperature(agent_id),
                    c_puct=1.2 + 0.1 * (agent_id % 5),
                    entropy_coef=0.002 + 0.001 * (agent_id % 3),
                )
            )
            self.replay[agent_id] = []

    def initialize_random(self):
        self.slots = []
        self.replay = {}
        for agent_id in range(self.population_size):
            dst = self.population_dir / f"agent_{agent_id}.pth"
            temperature = self._initial_temperature(agent_id)
            save_checkpoint(
                dst,
                PolicyMLP().to(self.device),
                agent_id=agent_id,
                parent_id=None,
                initialization="random",
                temperature=temperature,
            )
            self.slots.append(
                AgentSlot(
                    agent_id=agent_id,
                    checkpoint=dst,
                    parent_id=None,
                    temperature=temperature,
                    c_puct=1.2 + 0.1 * (agent_id % 5),
                    entropy_coef=0.002 + 0.001 * (agent_id % 3),
                    metadata={"initialization": "random"},
                )
            )
            self.replay[agent_id] = []

    def train_selfplay_games(
        self,
        num_games,
        sims=32,
        updates_per_game=1,
        heuristic_mix=0.25,
        random_mix=0.10,
        search_deadline_s=0.5,
        progress=False,
        desc=None,
    ):
        iterator = range(int(num_games))
        if progress:
            iterator = tqdm(iterator, desc=desc or "MCS self-play games", unit="game", leave=False)
        for game_idx in iterator:
            seats = self.sample_matchup()
            transitions = self._play_training_game(
                seats,
                sims=sims,
                seed=self.rng.randrange(10**9),
                heuristic_mix=heuristic_mix,
                random_mix=random_mix,
                search_deadline_s=search_deadline_s,
            )
            for agent_id, rows in transitions.items():
                self.replay.setdefault(agent_id, []).extend(rows)
                if len(self.replay[agent_id]) > self.replay_limit:
                    self.replay[agent_id] = self.replay[agent_id][-self.replay_limit :]
            losses = []
            for _ in range(int(updates_per_game)):
                for slot in self.slots:
                    loss = self._update_agent(slot)
                    if loss is not None:
                        losses.append(loss)
            if progress:
                replay_rows = sum(len(rows) for rows in self.replay.values())
                avg_loss = sum(losses) / len(losses) if losses else None
                postfix = {"matchup": ",".join(str(agent_id) for agent_id in seats), "replay": replay_rows}
                if avg_loss is not None:
                    postfix["loss"] = f"{avg_loss:.4f}"
                iterator.set_postfix(postfix)

    def sample_matchup(self):
        roll = self.rng.random()
        ids = [slot.agent_id for slot in self.slots]
        if roll < 0.20 and len(self.slots) >= 4:
            top = sorted(self.slots, key=lambda slot: slot.ema_points, reverse=True)[:2]
            selected = [self.rng.choice(top).agent_id]
            selected.extend(self.rng.sample([agent_id for agent_id in ids if agent_id not in selected], 3))
            return selected
        if roll < 0.30 and len(self.slots) >= 4:
            by_parent = {}
            for slot in self.slots:
                by_parent.setdefault(slot.parent_id if slot.parent_id is not None else slot.agent_id, []).append(slot.agent_id)
            selected = []
            for _, group in sorted(by_parent.items(), key=lambda item: len(item[1])):
                if group and len(selected) < 4:
                    selected.append(self.rng.choice(group))
            while len(selected) < 4:
                candidate = self.rng.choice(ids)
                if candidate not in selected:
                    selected.append(candidate)
            return selected
        return self.rng.sample(ids, 4)

    def evaluate_frozen(
        self,
        partitions=16,
        sims=8,
        heuristic_mix=0.25,
        random_mix=0.10,
        search_deadline_s=0.5,
        progress=False,
        desc=None,
    ):
        stats = {slot.agent_id: {"rank": 0.0, "points": 0.0, "score": 0.0, "games": 0} for slot in self.slots}
        pairwise = {slot.agent_id: {other.agent_id: 0.0 for other in self.slots} for slot in self.slots}
        ids = [slot.agent_id for slot in self.slots]
        iterator = range(int(partitions))
        if progress:
            iterator = tqdm(iterator, desc=desc or "Frozen eval partitions", unit="partition", leave=False)
        for partition_idx in iterator:
            shuffled = ids[:]
            self.rng.shuffle(shuffled)
            for table in [shuffled[i : i + 4] for i in range(0, len(shuffled), 4)]:
                if len(table) < 4:
                    continue
                base_state = deal_light_game(seed=self.rng.randrange(10**9))
                base_hands = [hand[:] for hand in base_state.hands]
                for shift in range(4):
                    state = base_state.clone()
                    state.hands = [base_hands[(seat + shift) % 4][:] for seat in range(4)]
                    self._play_eval_game(
                        state,
                        table,
                        sims=sims,
                        heuristic_mix=heuristic_mix,
                        random_mix=random_mix,
                        search_deadline_s=search_deadline_s,
                    )
                    ranks = state.ranks()
                    for seat, agent_id in enumerate(table):
                        stats[agent_id]["rank"] += ranks[seat]
                        stats[agent_id]["points"] += 4.5 - ranks[seat]
                        stats[agent_id]["score"] += state.scores[seat]
                        stats[agent_id]["games"] += 1
                    for i, j in itertools.permutations(range(4), 2):
                        if ranks[i] < ranks[j]:
                            pairwise[table[i]][table[j]] += 1.0
                        elif ranks[i] == ranks[j]:
                            pairwise[table[i]][table[j]] += 0.5
            if progress:
                games_so_far = sum(raw["games"] for raw in stats.values())
                iterator.set_postfix(games=games_so_far)

        records = {}
        for slot in self.slots:
            raw = stats[slot.agent_id]
            games = max(1, raw["games"])
            record = EvalRecord(
                avg_rank=raw["rank"] / games,
                avg_points=raw["points"] / games,
                avg_score=raw["score"] / games,
                games=raw["games"],
            )
            records[slot.agent_id] = record
            slot.ema_points = 0.75 * slot.ema_points + 0.25 * record.avg_points
            slot.ema_rank = 0.75 * slot.ema_rank + 0.25 * record.avg_rank

        elos = compute_bradley_terry_elo(pairwise, ids)
        for slot in self.slots:
            slot.elo = elos.get(slot.agent_id, slot.elo)
        return records

    def replace_bottom(self, generation, replace_count=1):
        ranked = sorted(self.slots, key=lambda slot: (slot.ema_points, -slot.ema_rank))
        elites = set(slot.agent_id for slot in sorted(self.slots, key=lambda slot: slot.ema_points, reverse=True)[:2])
        replaceable = [slot for slot in ranked if slot.agent_id not in elites and slot.grace <= 0]
        if not replaceable:
            return []
        top_sources = sorted(self.slots, key=lambda slot: slot.ema_points, reverse=True)[:3]
        replacements = []
        for victim in replaceable[: int(replace_count)]:
            source = top_sources[0] if self.rng.random() < 0.70 else self.rng.choice(top_sources)
            clone_checkpoint(
                source.checkpoint,
                victim.checkpoint,
                {"parent_id": source.agent_id, "generation": generation, "clone_reason": "bottom_replacement"},
            )
            victim.parent_id = source.agent_id
            victim.generation = generation
            victim.temperature = self._mutated_temperature(source.temperature)
            victim.c_puct = self._mutated_c_puct(source.c_puct)
            victim.entropy_coef = max(0.001, min(0.010, source.entropy_coef * self.rng.uniform(0.8, 1.3)))
            victim.ema_points = source.ema_points * 0.95
            victim.ema_rank = source.ema_rank
            victim.grace = 1
            victim.bottom_streak = 0
            self.replay[victim.agent_id] = []
            replacements.append((victim.agent_id, source.agent_id))
        for slot in self.slots:
            if slot.grace > 0:
                slot.grace -= 1
        return replacements

    def save_archive(self, generation, top_k=2):
        archive_dir = self.population_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        top = sorted(self.slots, key=lambda slot: slot.ema_points, reverse=True)[: int(top_k)]
        saved = []
        for slot in top:
            dst = archive_dir / f"gen_{generation}_agent_{slot.agent_id}.pth"
            clone_checkpoint(slot.checkpoint, dst, {"archive_generation": generation})
            saved.append(dst)
        return saved

    def _play_training_game(
        self,
        agent_ids,
        sims,
        seed,
        heuristic_mix,
        random_mix,
        search_deadline_s,
    ):
        state = deal_light_game(seed=seed)
        transitions = {agent_id: [] for agent_id in agent_ids}
        model_cache = {}
        for agent_id in agent_ids:
            model, _, _ = load_checkpoint(self._slot(agent_id).checkpoint, device=self.device)
            model.eval()
            model_cache[agent_id] = model
        while not state.is_terminal():
            history = state.history_for_player()
            actions = []
            for seat, agent_id in enumerate(agent_ids):
                if not state.hands[seat]:
                    continue
                slot = self._slot(agent_id)
                model = model_cache[agent_id]
                cfg = MCSConfig(
                    sims=sims,
                    c_puct=slot.c_puct,
                    root_temperature=slot.temperature,
                    heuristic_mix=heuristic_mix,
                    random_mix=random_mix,
                    deadline_s=search_deadline_s,
                )
                action, target_pi, _ = root_mcs_search(model, state.hands[seat], history, seat, cfg, self.rng)
                features, legal_mask = extract_policy_features(state.hands[seat], history, seat)
                transitions[agent_id].append((features, legal_mask, target_pi))
                actions.append((seat, action))
            state.step(actions)
        return transitions

    def _play_eval_game(
        self,
        state,
        agent_ids,
        sims,
        heuristic_mix,
        random_mix,
        search_deadline_s,
    ):
        model_cache = {}
        for agent_id in agent_ids:
            model, _, _ = load_checkpoint(self._slot(agent_id).checkpoint, device=self.device)
            model.eval()
            model_cache[agent_id] = model
        while not state.is_terminal():
            history = state.history_for_player()
            actions = []
            for seat, agent_id in enumerate(agent_ids):
                if not state.hands[seat]:
                    continue
                slot = self._slot(agent_id)
                model = model_cache[agent_id]
                cfg = MCSConfig(
                    sims=sims,
                    c_puct=slot.c_puct,
                    root_temperature=0.0,
                    rollout_temperature=0.7,
                    heuristic_mix=heuristic_mix,
                    random_mix=random_mix,
                    deadline_s=search_deadline_s,
                )
                action, _, _ = root_mcs_search(model, state.hands[seat], history, seat, cfg, self.rng)
                actions.append((seat, action))
            state.step(actions)

    def _update_agent(self, slot):
        rows = self.replay.get(slot.agent_id, [])
        if len(rows) < self.batch_size:
            return None
        model, optimizer_state, _ = load_checkpoint(slot.checkpoint, device=self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1.0e-4)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        batch = self.rng.sample(rows, self.batch_size)
        features = torch.tensor([row[0] for row in batch], dtype=torch.float32, device=self.device)
        legal = torch.tensor([row[1] for row in batch], dtype=torch.bool, device=self.device)
        targets = torch.tensor([row[2] for row in batch], dtype=torch.float32, device=self.device)
        logits = model(features)
        loss, _ = policy_loss(logits, legal, targets, entropy_coef=slot.entropy_coef)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        save_checkpoint(slot.checkpoint, model, optimizer, agent_id=slot.agent_id, parent_id=slot.parent_id)
        return float(loss.detach().cpu())

    def _slot(self, agent_id):
        for slot in self.slots:
            if slot.agent_id == agent_id:
                return slot
        raise KeyError(agent_id)

    def _initial_temperature(self, agent_id):
        return [1.20, 1.05, 0.95, 0.85][agent_id % 4]

    def _mutated_temperature(self, value):
        return max(0.65, min(1.35, value * self.rng.uniform(0.85, 1.20)))

    def _mutated_c_puct(self, value):
        return max(0.8, min(2.0, value * self.rng.uniform(0.85, 1.15)))


def compute_bradley_terry_elo(pairwise, ids, iters=80):
    strength = {agent_id: 1.0 for agent_id in ids}
    for _ in range(int(iters)):
        new_strength = {}
        for i in ids:
            wins = sum(pairwise[i].get(j, 0.0) for j in ids if j != i)
            denom = 0.0
            for j in ids:
                if i == j:
                    continue
                games = pairwise[i].get(j, 0.0) + pairwise[j].get(i, 0.0)
                if games > 0:
                    denom += games / max(1.0e-8, strength[i] + strength[j])
            new_strength[i] = wins / denom if denom > 0 else strength[i]
        avg = sum(new_strength.values()) / max(1, len(new_strength))
        if avg > 0:
            strength = {agent_id: value / avg for agent_id, value in new_strength.items()}
    raw = {agent_id: 400.0 * torch.log10(torch.tensor(max(value, 1.0e-8))).item() for agent_id, value in strength.items()}
    shift = 1500.0 - sum(raw.values()) / max(1, len(raw))
    return {agent_id: value + shift for agent_id, value in raw.items()}
