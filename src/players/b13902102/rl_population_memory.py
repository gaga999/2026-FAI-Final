import itertools
import multiprocessing as mp
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .rl_checkpoint import load_checkpoint, save_checkpoint
from .rl_features import action_index, extract_policy_features
from .rl_mcs import MCSConfig, root_mcs_search
from .rl_model import PolicyMLP, policy_loss
from .rl_population import AgentSlot, EvalRecord, compute_bradley_terry_elo
from .rl_train_env import deal_light_game


_WORKER_MODELS = {}


def _save_checkpoint_atomic(path, model, optimizer=None, **metadata):
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    save_checkpoint(temporary_path, model, optimizer, **metadata)
    temporary_path.replace(path)


def _worker_init(shared_models, limit_torch_threads=True):
    global _WORKER_MODELS
    if limit_torch_threads:
        torch.set_num_threads(1)
    _WORKER_MODELS = {
        int(agent_id): model
        for agent_id, model in shared_models.items()
    }
    for model in _WORKER_MODELS.values():
        model.eval()


def _sample_root_action(target_pi, hand, temperature, exploration_mix, rng):
    cards = sorted(int(card) for card in hand)
    if len(cards) == 1:
        return cards[0]

    temperature = max(float(temperature), 1.0e-6)
    weights = np.asarray(
        [
            max(float(target_pi[action_index(card)]), 1.0e-12)
            ** (1.0 / temperature)
            for card in cards
        ],
        dtype=np.float64,
    )
    weights /= max(float(weights.sum()), 1.0e-12)
    exploration_mix = min(1.0, max(0.0, float(exploration_mix)))
    if exploration_mix > 0.0:
        weights = (1.0 - exploration_mix) * weights
        weights += exploration_mix / len(cards)

    pick = rng.random()
    cumulative = 0.0
    for card, weight in zip(cards, weights):
        cumulative += float(weight)
        if pick <= cumulative:
            return card
    return cards[-1]


def _collect_training_game(task):
    (
        seed,
        agent_ids,
        slot_configs,
        sims,
        exploration_mix,
        heuristic_mix,
        random_mix,
        search_deadline_s,
    ) = task
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    state = deal_light_game(seed=rng.randrange(2**31))
    transitions = {int(agent_id): [] for agent_id in agent_ids}

    while not state.is_terminal():
        history = state.history_for_player()
        actions = []
        for seat, agent_id in enumerate(agent_ids):
            hand = state.hands[seat]
            if not hand:
                continue
            agent_id = int(agent_id)
            config = slot_configs[agent_id]
            search_config = MCSConfig(
                sims=int(sims),
                c_puct=float(config["c_puct"]),
                prior_temperature=1.0,
                rollout_temperature=1.0,
                root_temperature=0.0,
                heuristic_mix=float(heuristic_mix),
                random_mix=float(random_mix),
                deadline_s=search_deadline_s,
            )
            _, target_pi, _ = root_mcs_search(
                _WORKER_MODELS[agent_id],
                hand,
                history,
                seat,
                search_config,
                rng,
            )
            action = _sample_root_action(
                target_pi,
                hand,
                config["temperature"],
                exploration_mix,
                rng,
            )
            features, legal_mask = extract_policy_features(hand, history, seat)
            transitions[agent_id].append((features, legal_mask, target_pi))
            actions.append((seat, action))
        state.step(actions)

    return transitions


def _collect_eval_table(task):
    (
        seed,
        agent_ids,
        slot_configs,
        sims,
        heuristic_mix,
        random_mix,
        search_deadline_s,
    ) = task
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    base_state = deal_light_game(seed=rng.randrange(2**31))
    base_hands = [hand[:] for hand in base_state.hands]
    results = []

    for shift in range(4):
        state = base_state.clone()
        state.hands = [
            base_hands[(seat + shift) % 4][:]
            for seat in range(4)
        ]
        while not state.is_terminal():
            history = state.history_for_player()
            actions = []
            for seat, agent_id in enumerate(agent_ids):
                hand = state.hands[seat]
                if not hand:
                    continue
                agent_id = int(agent_id)
                config = slot_configs[agent_id]
                search_config = MCSConfig(
                    sims=int(sims),
                    c_puct=float(config["c_puct"]),
                    prior_temperature=1.0,
                    rollout_temperature=0.7,
                    root_temperature=0.0,
                    heuristic_mix=float(heuristic_mix),
                    random_mix=float(random_mix),
                    deadline_s=search_deadline_s,
                )
                action, _, _ = root_mcs_search(
                    _WORKER_MODELS[agent_id],
                    hand,
                    history,
                    seat,
                    search_config,
                    rng,
                )
                actions.append((seat, action))
            state.step(actions)
        results.append((tuple(agent_ids), state.ranks(), list(state.scores)))
    return results


class MemoryPopulationTrainer:
    def __init__(
        self,
        population_dir,
        device="cpu",
        population_size=8,
        seed=0,
        lr=3.0e-4,
        batch_size=256,
        replay_limit=6000,
        entropy_coef=0.003,
    ):
        self.population_dir = Path(population_dir)
        self.population_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.population_dir / "population_state.pt"
        self.device = device
        self.population_size = int(population_size)
        self.rng = random.Random(int(seed))
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.replay_limit = int(replay_limit)
        self.base_entropy_coef = float(entropy_coef)
        self.slots = []
        self.models = {}
        self.optimizers = {}
        self.replay = {}
        self.total_games = 0
        self.generation = 0
        self.worker_models = {}
        self.worker_pool = None
        self.worker_count = 0

    def initialize_from_checkpoint(self, checkpoint_path):
        self.close_workers()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
        base_model, _, _ = load_checkpoint(checkpoint_path, device=self.device)
        base_state = base_model.state_dict()
        self.slots = []
        self.models = {}
        self.optimizers = {}
        self.replay = {}
        self.total_games = 0
        self.generation = 0

        entropy_scales = (0.60, 0.80, 1.00, 1.30, 1.50, 1.20, 0.90, 0.70)
        for agent_id in range(self.population_size):
            checkpoint = self.population_dir / f"agent_{agent_id}.pth"
            model = PolicyMLP().to(self.device)
            model.load_state_dict(base_state)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=1.0e-4,
            )
            self.models[agent_id] = model
            self.optimizers[agent_id] = optimizer
            self.replay[agent_id] = []
            self.slots.append(
                AgentSlot(
                    agent_id=agent_id,
                    checkpoint=checkpoint,
                    temperature=self._initial_temperature(agent_id),
                    c_puct=1.2 + 0.1 * (agent_id % 5),
                    entropy_coef=self.base_entropy_coef
                    * entropy_scales[agent_id % len(entropy_scales)],
                    metadata={
                        "initialization": "checkpoint",
                        "init_checkpoint": str(checkpoint_path),
                    },
                )
            )

    def initialize_random(self):
        self.close_workers()
        self.slots = []
        self.models = {}
        self.optimizers = {}
        self.replay = {}
        self.total_games = 0
        self.generation = 0

        entropy_scales = (0.60, 0.80, 1.00, 1.30, 1.50, 1.20, 0.90, 0.70)
        for agent_id in range(self.population_size):
            checkpoint = self.population_dir / f"agent_{agent_id}.pth"
            model = PolicyMLP().to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=1.0e-4,
            )
            self.models[agent_id] = model
            self.optimizers[agent_id] = optimizer
            self.replay[agent_id] = []
            self.slots.append(
                AgentSlot(
                    agent_id=agent_id,
                    checkpoint=checkpoint,
                    temperature=self._initial_temperature(agent_id),
                    c_puct=1.2 + 0.1 * (agent_id % 5),
                    entropy_coef=self.base_entropy_coef
                    * entropy_scales[agent_id % len(entropy_scales)],
                    metadata={"initialization": "random"},
                )
            )

    def resume(self):
        self.close_workers()
        if not self.state_path.exists():
            raise FileNotFoundError(f"Population state not found: {self.state_path}")
        payload = torch.load(self.state_path, map_location="cpu")
        self.total_games = int(payload.get("total_games", 0))
        self.generation = int(payload.get("generation", 0))
        if payload.get("rng_state") is not None:
            self.rng.setstate(payload["rng_state"])
        self.slots = []
        self.models = {}
        self.optimizers = {}
        self.replay = {}

        for raw_slot in payload["slots"]:
            raw_slot = dict(raw_slot)
            raw_slot["checkpoint"] = Path(raw_slot["checkpoint"])
            slot = AgentSlot(**raw_slot)
            if not slot.checkpoint.exists():
                raise FileNotFoundError(
                    f"Agent checkpoint not found during resume: {slot.checkpoint}"
                )
            model, optimizer_state, _ = load_checkpoint(
                slot.checkpoint,
                device=self.device,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=1.0e-4,
            )
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = self.lr
            self.slots.append(slot)
            self.models[slot.agent_id] = model
            self.optimizers[slot.agent_id] = optimizer
            self.replay[slot.agent_id] = []

    def collect_training_games(
        self,
        num_games,
        sims=32,
        num_workers=1,
        exploration_mix=0.03,
        heuristic_mix=0.25,
        random_mix=0.10,
        task_chunksize=1,
        search_deadline_s=None,
        progress=True,
        desc=None,
    ):
        tasks = []
        for _ in range(int(num_games)):
            agent_ids = tuple(self.sample_matchup())
            tasks.append(
                (
                    self.rng.randrange(2**31),
                    agent_ids,
                    self._slot_configs(agent_ids),
                    int(sims),
                    float(exploration_mix),
                    float(heuristic_mix),
                    float(random_mix),
                    search_deadline_s,
                )
            )
        results = self._run_worker_tasks(
            _collect_training_game,
            tasks,
            num_workers,
            task_chunksize,
        )
        iterator = tqdm(
            results,
            total=len(tasks),
            desc=desc or "MCS collect",
            unit="game",
            leave=False,
            disable=not progress,
        )
        collected_rows = 0
        for transitions in iterator:
            for agent_id, rows in transitions.items():
                replay = self.replay[int(agent_id)]
                replay.extend(rows)
                if len(replay) > self.replay_limit:
                    del replay[: len(replay) - self.replay_limit]
                collected_rows += len(rows)
            self.total_games += 1
            iterator.set_postfix(
                rows=collected_rows,
                replay=sum(len(rows) for rows in self.replay.values()),
            )
        return collected_rows

    def update_population(self, update_steps_per_agent=96, progress=True, desc=None):
        iterator = self.slots
        if progress:
            iterator = tqdm(
                self.slots,
                desc=desc or "MCS updates",
                unit="agent",
                leave=False,
            )
        metrics = {}
        for slot in iterator:
            losses = []
            entropies = []
            for _ in range(int(update_steps_per_agent)):
                result = self._update_agent(slot)
                if result is None:
                    break
                losses.append(result["loss"])
                entropies.append(result["entropy"])
            metrics[slot.agent_id] = {
                "loss": sum(losses) / max(1, len(losses)),
                "entropy": sum(entropies) / max(1, len(entropies)),
                "updates": len(losses),
            }
            if progress:
                iterator.set_postfix(
                    agent=slot.agent_id,
                    loss=f"{metrics[slot.agent_id]['loss']:.4f}",
                    entropy=f"{metrics[slot.agent_id]['entropy']:.3f}",
                )
        return metrics

    def evaluate_frozen(
        self,
        partitions=16,
        sims=8,
        num_workers=1,
        heuristic_mix=0.25,
        random_mix=0.10,
        task_chunksize=1,
        search_deadline_s=None,
        progress=True,
        desc=None,
    ):
        stats = {
            slot.agent_id: {
                "rank": 0.0,
                "points": 0.0,
                "score": 0.0,
                "games": 0,
            }
            for slot in self.slots
        }
        pairwise = {
            slot.agent_id: {
                other.agent_id: 0.0
                for other in self.slots
            }
            for slot in self.slots
        }
        tasks = []
        ids = [slot.agent_id for slot in self.slots]
        for _ in range(int(partitions)):
            shuffled = ids[:]
            self.rng.shuffle(shuffled)
            for start in range(0, len(shuffled), 4):
                table = shuffled[start : start + 4]
                if len(table) == 4:
                    tasks.append(
                        (
                            self.rng.randrange(2**31),
                            tuple(table),
                            self._slot_configs(table),
                            int(sims),
                            float(heuristic_mix),
                            float(random_mix),
                            search_deadline_s,
                        )
                    )

        results = self._run_worker_tasks(
            _collect_eval_table,
            tasks,
            num_workers,
            task_chunksize,
        )
        iterator = tqdm(
            results,
            total=len(tasks),
            desc=desc or "MCS frozen eval",
            unit="table",
            leave=False,
            disable=not progress,
        )
        for table_results in iterator:
            for agent_ids, ranks, scores in table_results:
                for seat, agent_id in enumerate(agent_ids):
                    stats[agent_id]["rank"] += ranks[seat]
                    stats[agent_id]["points"] += 4.5 - ranks[seat]
                    stats[agent_id]["score"] += scores[seat]
                    stats[agent_id]["games"] += 1
                for i, j in itertools.permutations(range(4), 2):
                    if ranks[i] < ranks[j]:
                        pairwise[agent_ids[i]][agent_ids[j]] += 1.0
                    elif ranks[i] == ranks[j]:
                        pairwise[agent_ids[i]][agent_ids[j]] += 0.5
            iterator.set_postfix(
                games=sum(raw["games"] for raw in stats.values())
            )

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
        ranked = sorted(
            self.slots,
            key=lambda slot: (slot.ema_points, -slot.ema_rank),
        )
        elites = {
            slot.agent_id
            for slot in sorted(
                self.slots,
                key=lambda slot: slot.ema_points,
                reverse=True,
            )[:2]
        }
        replaceable = [
            slot
            for slot in ranked
            if slot.agent_id not in elites and slot.grace <= 0
        ]
        top_sources = sorted(
            self.slots,
            key=lambda slot: slot.ema_points,
            reverse=True,
        )[:3]
        replacements = []
        replaced_ids = set()

        for victim in replaceable[: int(replace_count)]:
            source = (
                top_sources[0]
                if self.rng.random() < 0.70
                else self.rng.choice(top_sources)
            )
            self.models[victim.agent_id].load_state_dict(
                self.models[source.agent_id].state_dict()
            )
            self.optimizers[victim.agent_id] = torch.optim.AdamW(
                self.models[victim.agent_id].parameters(),
                lr=self.lr,
                weight_decay=1.0e-4,
            )
            victim.parent_id = source.agent_id
            victim.generation = int(generation)
            victim.temperature = self._mutated_temperature(source.temperature)
            victim.c_puct = self._mutated_c_puct(source.c_puct)
            victim.entropy_coef = max(
                0.001,
                min(
                    0.010,
                    source.entropy_coef * self.rng.uniform(0.8, 1.3),
                ),
            )
            victim.ema_points = source.ema_points * 0.95
            victim.ema_rank = source.ema_rank
            victim.grace = 1
            victim.bottom_streak = 0
            self.replay[victim.agent_id] = []
            replacements.append((victim.agent_id, source.agent_id))
            replaced_ids.add(victim.agent_id)

        for slot in self.slots:
            if slot.grace > 0 and slot.agent_id not in replaced_ids:
                slot.grace -= 1
        return replacements

    def save_population(self, generation=None, reason="periodic"):
        if generation is not None:
            self.generation = int(generation)
        for slot in self.slots:
            _save_checkpoint_atomic(
                slot.checkpoint,
                self.models[slot.agent_id],
                self.optimizers[slot.agent_id],
                stage="mcs_memory",
                reason=reason,
                generation=self.generation,
                total_games=self.total_games,
                agent_id=slot.agent_id,
                parent_id=slot.parent_id,
                temperature=slot.temperature,
                c_puct=slot.c_puct,
                entropy_coef=slot.entropy_coef,
                initialization=slot.metadata.get("initialization"),
                init_checkpoint=slot.metadata.get("init_checkpoint"),
            )
        state_payload = {
            "generation": self.generation,
            "total_games": self.total_games,
            "rng_state": self.rng.getstate(),
            "slots": [
                {
                    **asdict(slot),
                    "checkpoint": str(slot.checkpoint),
                }
                for slot in self.slots
            ],
            "replay_persisted": False,
        }
        temporary_path = self.state_path.with_name(
            f".{self.state_path.name}.tmp"
        )
        torch.save(state_payload, temporary_path)
        temporary_path.replace(self.state_path)

    def save_archive(self, generation, top_k=2):
        archive_dir = self.population_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        top = sorted(
            self.slots,
            key=lambda slot: slot.ema_points,
            reverse=True,
        )[: int(top_k)]
        saved = []
        for rank, slot in enumerate(top):
            destination = (
                archive_dir
                / f"gen_{int(generation)}_agent_{slot.agent_id}.pth"
            )
            _save_checkpoint_atomic(
                destination,
                self.models[slot.agent_id],
                stage="mcs_memory_archive",
                generation=int(generation),
                total_games=self.total_games,
                agent_id=slot.agent_id,
                avg_points=slot.ema_points,
                avg_rank=slot.ema_rank,
                elo=slot.elo,
            )
            saved.append(destination)
            if rank == 0:
                _save_checkpoint_atomic(
                    archive_dir / "best.pth",
                    self.models[slot.agent_id],
                    stage="mcs_memory_best",
                    generation=int(generation),
                    total_games=self.total_games,
                    agent_id=slot.agent_id,
                    avg_points=slot.ema_points,
                    avg_rank=slot.ema_rank,
                    elo=slot.elo,
                )
        return saved

    def sample_matchup(self):
        roll = self.rng.random()
        ids = [slot.agent_id for slot in self.slots]
        if roll < 0.20 and len(self.slots) >= 4:
            top = sorted(
                self.slots,
                key=lambda slot: slot.ema_points,
                reverse=True,
            )[:2]
            selected = [self.rng.choice(top).agent_id]
            selected.extend(
                self.rng.sample(
                    [
                        agent_id
                        for agent_id in ids
                        if agent_id not in selected
                    ],
                    3,
                )
            )
            return selected
        if roll < 0.30 and len(self.slots) >= 4:
            by_parent = {}
            for slot in self.slots:
                lineage = (
                    slot.parent_id
                    if slot.parent_id is not None
                    else slot.agent_id
                )
                by_parent.setdefault(lineage, []).append(slot.agent_id)
            selected = []
            for _, group in sorted(
                by_parent.items(),
                key=lambda item: len(item[1]),
            ):
                if group and len(selected) < 4:
                    selected.append(self.rng.choice(group))
            while len(selected) < 4:
                candidate = self.rng.choice(ids)
                if candidate not in selected:
                    selected.append(candidate)
            return selected
        return self.rng.sample(ids, 4)

    def set_learning_rate(self, learning_rate):
        self.lr = float(learning_rate)
        for optimizer in self.optimizers.values():
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = self.lr

    def _update_agent(self, slot):
        rows = self.replay.get(slot.agent_id, [])
        if len(rows) < self.batch_size:
            return None
        batch = self.rng.sample(rows, self.batch_size)
        features = torch.from_numpy(
            np.stack([row[0] for row in batch])
        ).float().to(self.device)
        legal_masks = torch.from_numpy(
            np.stack([row[1] for row in batch])
        ).bool().to(self.device)
        targets = torch.from_numpy(
            np.stack([row[2] for row in batch])
        ).float().to(self.device)

        model = self.models[slot.agent_id]
        optimizer = self.optimizers[slot.agent_id]
        model.train()
        logits = model(features)
        loss, loss_metrics = policy_loss(
            logits,
            legal_masks,
            targets,
            entropy_coef=slot.entropy_coef,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "entropy": float(loss_metrics["entropy"].cpu()),
        }

    def _slot_configs(self, agent_ids):
        return {
            slot.agent_id: {
                "temperature": slot.temperature,
                "c_puct": slot.c_puct,
            }
            for slot in self.slots
            if slot.agent_id in agent_ids
        }

    def _make_shared_worker_models(self):
        self.worker_models = {}
        for agent_id, source_model in self.models.items():
            shared_model = PolicyMLP().cpu()
            shared_model.share_memory()
            self.worker_models[agent_id] = shared_model
            self._copy_to_worker_model(source_model, shared_model)

    def _copy_to_worker_model(self, source_model, worker_model):
        source_state = source_model.state_dict()
        worker_state = worker_model.state_dict()
        with torch.no_grad():
            for name, destination in worker_state.items():
                destination.copy_(source_state[name].detach().cpu())
        worker_model.eval()

    def _sync_worker_models(self):
        if set(self.worker_models) != set(self.models):
            self._make_shared_worker_models()
            return
        for agent_id, source_model in self.models.items():
            self._copy_to_worker_model(
                source_model,
                self.worker_models[agent_id],
            )

    def _ensure_workers(self, num_workers):
        num_workers = max(1, int(num_workers))
        if not self.worker_models:
            self._make_shared_worker_models()
        if self.worker_pool is not None and self.worker_count != num_workers:
            self.close_workers()
            self._make_shared_worker_models()
        if num_workers == 1:
            _worker_init(self.worker_models, limit_torch_threads=False)
            self.worker_count = 1
            return
        if self.worker_pool is None:
            context = mp.get_context("spawn")
            self.worker_pool = context.Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(self.worker_models, True),
            )
            self.worker_count = num_workers

    def close_workers(self):
        if self.worker_pool is not None:
            self.worker_pool.close()
            self.worker_pool.join()
        self.worker_pool = None
        self.worker_count = 0
        self.worker_models = {}

    def _run_worker_tasks(
        self,
        function,
        tasks,
        num_workers,
        chunksize=1,
    ):
        num_workers = max(1, int(num_workers))
        self._ensure_workers(num_workers)
        self._sync_worker_models()
        if num_workers == 1:
            return map(function, tasks)

        chunksize = max(1, int(chunksize))
        return self.worker_pool.imap_unordered(
            function,
            tasks,
            chunksize=chunksize,
        )

    def _initial_temperature(self, agent_id):
        return (1.25, 1.10, 0.95, 0.80)[agent_id % 4]

    def _mutated_temperature(self, value):
        return max(
            0.65,
            min(1.35, value * self.rng.uniform(0.85, 1.20)),
        )

    def _mutated_c_puct(self, value):
        return max(
            0.8,
            min(2.0, value * self.rng.uniform(0.85, 1.15)),
        )
