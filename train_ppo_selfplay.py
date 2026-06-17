import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from src.players.b13902102.rl_checkpoint import load_checkpoint, save_checkpoint
from src.players.b13902102.rl_features import action_index, extract_policy_features
from src.players.b13902102.rl_model import mask_logits
from src.players.b13902102.rl_train_env import deal_light_game


def run_episode(model, rng, temperature=1.0):
    state = deal_light_game(seed=rng.randrange(10**9))
    rows = []
    while not state.is_terminal():
        history = state.history_for_player()
        actions = []
        step_rows = []
        for seat, hand in enumerate(state.hands):
            if not hand:
                continue
            features, legal_mask = extract_policy_features(hand, history, seat)
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0)
                legal = torch.from_numpy(legal_mask).bool().unsqueeze(0)
                logits = mask_logits(model(x), legal).squeeze(0)
                probs = torch.softmax(logits / max(temperature, 1.0e-6), dim=-1)
                action_idx = int(torch.multinomial(probs, 1).item())
                old_logprob = float(torch.log(probs[action_idx].clamp_min(1.0e-8)).item())
            card = action_idx + 1
            if card not in hand:
                card = min(hand)
                action_idx = action_index(card)
            actions.append((seat, card))
            step_rows.append((features, legal_mask, action_idx, old_logprob, seat))
        state.step(actions)
        rows.extend(step_rows)
    returns = [state.utility(row[4]) for row in rows]
    return [(row[0], row[1], row[2], row[3], ret) for row, ret in zip(rows, returns)]


def train_ppo_lite(init_checkpoint, output_checkpoint, episodes=1000, batch_size=512, lr=1.0e-5, clip_ratio=0.15, entropy_coef=0.005, seed=0):
    rng = random.Random(seed)
    print(f"[PPO] Loading checkpoint: {init_checkpoint}", flush=True)
    model, _, _ = load_checkpoint(init_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    replay = []
    print(f"[PPO] Training PPO-lite: episodes={int(episodes)}, batch_size={int(batch_size)}, lr={float(lr)}", flush=True)
    episode_bar = tqdm(range(1, int(episodes) + 1), desc="PPO episodes", unit="episode")
    for episode in episode_bar:
        model.eval()
        replay.extend(run_episode(model, rng))
        if len(replay) < batch_size:
            episode_bar.set_postfix(replay=len(replay), status="collecting")
            continue
        batch = rng.sample(replay, batch_size)
        features = torch.tensor([row[0] for row in batch], dtype=torch.float32)
        legal = torch.tensor([row[1] for row in batch], dtype=torch.bool)
        actions = torch.tensor([row[2] for row in batch], dtype=torch.long)
        old_log_probs = torch.tensor([row[3] for row in batch], dtype=torch.float32)
        returns = torch.tensor([row[4] for row in batch], dtype=torch.float32)
        advantages = (returns - returns.mean()) / returns.std().clamp_min(1.0e-6)

        model.train()
        logits = mask_logits(model(features), legal)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(chosen - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        replay = replay[-20000:]
        episode_bar.set_postfix(replay=len(replay), loss=f"{float(loss.detach().cpu()):.4f}")
        if episode % 50 == 0:
            tqdm.write(f"[PPO] Episode {episode}/{int(episodes)}: loss={float(loss.detach().cpu()):.4f}, replay={len(replay)}")
    print(f"[PPO] Saving checkpoint: {output_checkpoint}", flush=True)
    save_checkpoint(output_checkpoint, model, optimizer, stage="ppo_lite", init_checkpoint=str(init_checkpoint))


def main():
    parser = argparse.ArgumentParser(description="Optional actor-only PPO-lite fine-tuning.")
    parser.add_argument("--init-checkpoint", type=str, default="artifacts/b13902102/population/archive/best.pth")
    parser.add_argument("--output-checkpoint", type=str, default="artifacts/b13902102/ppo_policy.pth")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_ppo_lite(args.init_checkpoint, args.output_checkpoint, episodes=args.episodes, seed=args.seed)
    print("[PPO] Done", flush=True)


if __name__ == "__main__":
    main()
