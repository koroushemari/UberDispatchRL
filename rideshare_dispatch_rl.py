from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

def _copy_network(network: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in network.items()}


def _init_mlp(
    rng: np.random.Generator,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    scale: float = 0.15,
) -> dict[str, np.ndarray]:
    return {
        "W1": rng.normal(0.0, scale, size=(input_dim, hidden_dim)).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "W2": rng.normal(0.0, scale, size=(hidden_dim, output_dim)).astype(np.float32),
        "b2": np.zeros(output_dim, dtype=np.float32),
    }


def _forward_mlp(network: dict[str, np.ndarray], inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_pre = inputs @ network["W1"] + network["b1"]
    hidden = np.maximum(0.0, hidden_pre)
    outputs = hidden @ network["W2"] + network["b2"]
    return hidden_pre, hidden, outputs


def _backward_mlp(
    network: dict[str, np.ndarray],
    inputs: np.ndarray,
    hidden_pre: np.ndarray,
    hidden: np.ndarray,
    output_grads: np.ndarray,
) -> dict[str, np.ndarray]:
    grad_W2 = hidden.T @ output_grads
    grad_b2 = output_grads.sum(axis=0)
    hidden_grads = output_grads @ network["W2"].T
    hidden_pre_grads = hidden_grads * (hidden_pre > 0.0)
    grad_W1 = inputs.T @ hidden_pre_grads
    grad_b1 = hidden_pre_grads.sum(axis=0)
    return {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }


def _apply_grads(
    network: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    learning_rate: float,
    clip_value: float = 1.0,
) -> None:
    for name in network:
        grad = np.clip(grads[name], -clip_value, clip_value).astype(np.float32)
        network[name] -= learning_rate * grad


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _action_mask(valid_actions, n_actions: int) -> np.ndarray:
    if valid_actions is None:
        return np.ones(n_actions, dtype=bool)
    mask = np.asarray(valid_actions, dtype=bool).reshape(-1)
    if mask.size != n_actions:
        raise ValueError(f"Expected action mask of length {n_actions}, got {mask.size}")
    if not np.any(mask):
        return np.ones(n_actions, dtype=bool)
    return mask


def _masked_argmax(values: np.ndarray, valid_actions) -> int:
    mask = _action_mask(valid_actions, values.shape[-1])
    masked_values = np.where(mask, values, -1e12)
    return int(np.argmax(masked_values))


def _sample_masked_action(rng: np.random.Generator, valid_actions, n_actions: int) -> int:
    mask = _action_mask(valid_actions, n_actions)
    choices = np.flatnonzero(mask)
    return int(rng.choice(choices))


@dataclass
class ReplayTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    next_action_mask: np.ndarray


class QLearningAgent:
    name = "q_learning"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        seed: int = 0,
        learning_rate: float = 0.15,
        discount: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2500,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / float(max(1, self.epsilon_decay_steps)))
        return float(self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start))

    def act(self, state_id: int, training: bool = True, valid_actions=None) -> int:
        if training and self.rng.random() < self.epsilon():
            return _sample_masked_action(self.rng, valid_actions, self.n_actions)
        return _masked_argmax(self.q_table[state_id], valid_actions)

    def update(self, state_id: int, action: int, reward: float, next_state_id: int, done: bool, next_valid_actions=None) -> None:
        next_value = 0.0 if done else float(np.max(np.where(_action_mask(next_valid_actions, self.n_actions), self.q_table[next_state_id], -1e12)))
        td_target = reward + self.discount * next_value
        self.q_table[state_id, action] += self.learning_rate * (td_target - self.q_table[state_id, action])
        self.steps += 1

    def snapshot(self) -> dict[str, np.ndarray | int]:
        return {"q_table": self.q_table.copy(), "steps": self.steps}

    def restore(self, snapshot: dict[str, np.ndarray | int]) -> None:
        self.q_table = snapshot["q_table"].copy()
        self.steps = int(snapshot["steps"])


class SARSAAgent:
    name = "sarsa"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        seed: int = 0,
        learning_rate: float = 0.15,
        discount: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2500,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / float(max(1, self.epsilon_decay_steps)))
        return float(self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start))

    def act(self, state_id: int, training: bool = True, valid_actions=None) -> int:
        if training and self.rng.random() < self.epsilon():
            return _sample_masked_action(self.rng, valid_actions, self.n_actions)
        return _masked_argmax(self.q_table[state_id], valid_actions)

    def update(
        self,
        state_id: int,
        action: int,
        reward: float,
        next_state_id: int,
        next_action: int,
        done: bool,
    ) -> None:
        next_value = 0.0 if done else float(self.q_table[next_state_id, next_action])
        td_target = reward + self.discount * next_value
        self.q_table[state_id, action] += self.learning_rate * (td_target - self.q_table[state_id, action])
        self.steps += 1

    def snapshot(self) -> dict[str, np.ndarray | int]:
        return {"q_table": self.q_table.copy(), "steps": self.steps}

    def restore(self, snapshot: dict[str, np.ndarray | int]) -> None:
        self.q_table = snapshot["q_table"].copy()
        self.steps = int(snapshot["steps"])


class DQNAgent:
    name = "dqn"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 0,
        hidden_dim: int = 48,
        learning_rate: float = 0.003,
        discount: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        buffer_size: int = 10000,
        batch_size: int = 64,
        warmup_steps: int = 300,
        target_sync_steps: int = 200,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_sync_steps = target_sync_steps
        self.rng = np.random.default_rng(seed)
        self.policy_net = _init_mlp(self.rng, obs_dim, hidden_dim, n_actions)
        self.target_net = _copy_network(self.policy_net)
        self.replay = deque(maxlen=buffer_size)
        self.steps = 0
        self.last_loss = 0.0

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / float(max(1, self.epsilon_decay_steps)))
        return float(self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start))

    def q_values(self, obs: np.ndarray, network: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if network is None:
            network = self.policy_net
        batch = np.atleast_2d(obs.astype(np.float32))
        _, _, outputs = _forward_mlp(network, batch)
        return outputs[0]

    def act(self, obs: np.ndarray, training: bool = True, valid_actions=None) -> int:
        if training and self.rng.random() < self.epsilon():
            return _sample_masked_action(self.rng, valid_actions, self.n_actions)
        return _masked_argmax(self.q_values(obs), valid_actions)

    def store(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool, next_action_mask=None) -> None:
        self.replay.append(
            ReplayTransition(
                obs=np.asarray(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
                next_action_mask=_action_mask(next_action_mask, self.n_actions).astype(bool),
            )
        )

    def update(self) -> float:
        self.steps += 1
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return 0.0

        batch_indices = self.rng.choice(len(self.replay), size=self.batch_size, replace=False)
        transitions = [self.replay[int(idx)] for idx in batch_indices]

        obs_batch = np.stack([transition.obs for transition in transitions]).astype(np.float32)
        action_batch = np.array([transition.action for transition in transitions], dtype=np.int64)
        reward_batch = np.array([transition.reward for transition in transitions], dtype=np.float32)
        next_obs_batch = np.stack([transition.next_obs for transition in transitions]).astype(np.float32)
        done_batch = np.array([transition.done for transition in transitions], dtype=np.float32)
        next_mask_batch = np.stack([transition.next_action_mask for transition in transitions]).astype(bool)

        hidden_pre, hidden, q_values = _forward_mlp(self.policy_net, obs_batch)
        _, _, next_q_values = _forward_mlp(self.target_net, next_obs_batch)
        chosen_q = q_values[np.arange(self.batch_size), action_batch]
        masked_next_q = np.where(next_mask_batch, next_q_values, -1e12)
        targets = reward_batch + self.discount * (1.0 - done_batch) * np.max(masked_next_q, axis=1)

        errors = chosen_q - targets
        output_grads = np.zeros_like(q_values, dtype=np.float32)
        output_grads[np.arange(self.batch_size), action_batch] = errors / float(self.batch_size)
        grads = _backward_mlp(self.policy_net, obs_batch, hidden_pre, hidden, output_grads)
        _apply_grads(self.policy_net, grads, self.learning_rate)
        self.last_loss = float(0.5 * np.mean(errors ** 2))

        if self.steps % self.target_sync_steps == 0:
            self.target_net = _copy_network(self.policy_net)

        return self.last_loss

    def snapshot(self) -> dict[str, object]:
        return {
            "policy": _copy_network(self.policy_net),
            "target": _copy_network(self.target_net),
            "steps": self.steps,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.policy_net = _copy_network(snapshot["policy"])
        self.target_net = _copy_network(snapshot["target"])
        self.steps = int(snapshot["steps"])


class PPOAgent:
    name = "ppo"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 0,
        hidden_dim: int = 48,
        policy_lr: float = 0.002,
        value_lr: float = 0.004,
        discount: float = 0.98,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        update_epochs: int = 6,
        minibatch_size: int = 64,
        rollout_episodes: int = 12,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_episodes = rollout_episodes
        self.rng = np.random.default_rng(seed)
        self.policy_net = _init_mlp(self.rng, obs_dim, hidden_dim, n_actions, scale=0.12)
        self.value_net = _init_mlp(self.rng, obs_dim, hidden_dim, 1, scale=0.12)
        self.trajectory_buffer: list[dict[str, np.ndarray]] = []
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0

    def _policy_outputs(self, obs_batch: np.ndarray, action_masks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hidden_pre, hidden, logits = _forward_mlp(self.policy_net, obs_batch)
        if action_masks is not None:
            safe_masks = np.asarray(action_masks, dtype=bool)
            if safe_masks.ndim == 1:
                safe_masks = np.repeat(safe_masks[None, :], len(obs_batch), axis=0)
            empty_rows = ~np.any(safe_masks, axis=1)
            if np.any(empty_rows):
                safe_masks[empty_rows] = True
            logits = np.where(safe_masks, logits, -1e12)
        probs = _softmax(logits)
        return hidden_pre, hidden, logits, probs

    def _value_outputs(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_pre, hidden, values = _forward_mlp(self.value_net, obs_batch)
        return hidden_pre, hidden, values[:, 0]

    def act(self, obs: np.ndarray, training: bool = True, valid_actions=None) -> tuple[int, float, float]:
        obs_batch = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        action_mask = _action_mask(valid_actions, self.n_actions)
        _, _, _, probs = self._policy_outputs(obs_batch, action_mask[None, :])
        _, _, values = self._value_outputs(obs_batch)
        probs = probs[0]
        if training:
            action = int(self.rng.choice(self.n_actions, p=probs))
        else:
            action = int(np.argmax(probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        value = float(values[0])
        return action, log_prob, value

    def remember_episode(
        self,
        observations: Iterable[np.ndarray],
        actions: Iterable[int],
        rewards: Iterable[float],
        log_probs: Iterable[float],
        values: Iterable[float],
        dones: Iterable[bool],
        action_masks: Iterable[np.ndarray],
    ) -> None:
        obs_array = np.asarray(list(observations), dtype=np.float32)
        actions_array = np.asarray(list(actions), dtype=np.int64)
        rewards_array = np.asarray(list(rewards), dtype=np.float32)
        log_prob_array = np.asarray(list(log_probs), dtype=np.float32)
        values_array = np.asarray(list(values), dtype=np.float32)
        dones_array = np.asarray(list(dones), dtype=np.float32)
        action_mask_array = np.asarray([_action_mask(mask, self.n_actions) for mask in action_masks], dtype=bool)

        advantages = np.zeros_like(rewards_array, dtype=np.float32)
        returns = np.zeros_like(rewards_array, dtype=np.float32)

        next_advantage = 0.0
        next_value = 0.0
        for index in reversed(range(len(rewards_array))):
            mask = 1.0 - dones_array[index]
            delta = rewards_array[index] + self.discount * next_value * mask - values_array[index]
            next_advantage = delta + self.discount * self.gae_lambda * mask * next_advantage
            advantages[index] = next_advantage
            returns[index] = advantages[index] + values_array[index]
            next_value = values_array[index]

        self.trajectory_buffer.append(
            {
                "observations": obs_array,
                "actions": actions_array,
                "old_log_probs": log_prob_array,
                "advantages": advantages,
                "returns": returns,
                "action_masks": action_mask_array,
            }
        )

    def should_update(self) -> bool:
        return len(self.trajectory_buffer) >= self.rollout_episodes

    def update(self) -> tuple[float, float]:
        if not self.trajectory_buffer:
            return 0.0, 0.0

        observations = np.concatenate([episode["observations"] for episode in self.trajectory_buffer], axis=0)
        actions = np.concatenate([episode["actions"] for episode in self.trajectory_buffer], axis=0)
        old_log_probs = np.concatenate([episode["old_log_probs"] for episode in self.trajectory_buffer], axis=0)
        advantages = np.concatenate([episode["advantages"] for episode in self.trajectory_buffer], axis=0)
        returns = np.concatenate([episode["returns"] for episode in self.trajectory_buffer], axis=0)
        action_masks = np.concatenate([episode["action_masks"] for episode in self.trajectory_buffer], axis=0)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0

        indices = np.arange(len(observations))
        batch_size = min(self.minibatch_size, len(observations))

        for _ in range(self.update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                obs_batch = observations[batch_indices]
                action_batch = actions[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]
                action_mask_batch = action_masks[batch_indices]

                hidden_pre, hidden, logits, probs = self._policy_outputs(obs_batch, action_mask_batch)
                chosen_probs = probs[np.arange(len(action_batch)), action_batch]
                new_log_probs = np.log(chosen_probs + 1e-8)
                ratios = np.exp(new_log_probs - old_log_prob_batch)

                active = np.logical_or(
                    np.logical_and(advantage_batch >= 0.0, ratios <= (1.0 + self.clip_epsilon)),
                    np.logical_and(advantage_batch < 0.0, ratios >= (1.0 - self.clip_epsilon)),
                )
                coeff = np.where(active, (advantage_batch * ratios) / float(len(action_batch)), 0.0)
                policy_grads_out = coeff[:, None] * probs
                policy_grads_out[np.arange(len(action_batch)), action_batch] -= coeff
                policy_grads = _backward_mlp(
                    self.policy_net,
                    obs_batch,
                    hidden_pre,
                    hidden,
                    policy_grads_out.astype(np.float32),
                )
                _apply_grads(self.policy_net, policy_grads, self.policy_lr)

                value_hidden_pre, value_hidden, values = self._value_outputs(obs_batch)
                value_errors = values - return_batch
                value_grads_out = (value_errors / float(len(action_batch)))[:, None]
                value_grads = _backward_mlp(
                    self.value_net,
                    obs_batch,
                    value_hidden_pre,
                    value_hidden,
                    value_grads_out.astype(np.float32),
                )
                _apply_grads(self.value_net, value_grads, self.value_lr)

                unclipped = ratios * advantage_batch
                clipped = np.clip(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_batch
                total_policy_loss += float(-np.mean(np.minimum(unclipped, clipped)))
                total_value_loss += float(0.5 * np.mean(value_errors ** 2))
                total_batches += 1

        self.trajectory_buffer.clear()
        self.last_policy_loss = total_policy_loss / float(max(1, total_batches))
        self.last_value_loss = total_value_loss / float(max(1, total_batches))
        return self.last_policy_loss, self.last_value_loss

    def snapshot(self) -> dict[str, object]:
        return {
            "policy": _copy_network(self.policy_net),
            "value": _copy_network(self.value_net),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.policy_net = _copy_network(snapshot["policy"])
        self.value_net = _copy_network(snapshot["value"])


class SparseQLearningAgent:
    name = "q_learning"

    def __init__(
        self,
        n_actions: int,
        seed: int = 0,
        learning_rate: float = 0.12,
        discount: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 6000,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.q_values: dict[object, np.ndarray] = {}

    def _row(self, state_key) -> np.ndarray:
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_values[state_key]

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / float(max(1, self.epsilon_decay_steps)))
        return float(self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start))

    def act(self, state_key, training: bool = True, valid_actions=None) -> int:
        if training and self.rng.random() < self.epsilon():
            return _sample_masked_action(self.rng, valid_actions, self.n_actions)
        return _masked_argmax(self._row(state_key), valid_actions)

    def update(self, state_key, action: int, reward: float, next_state_key, done: bool, next_valid_actions=None) -> None:
        current_row = self._row(state_key)
        next_row = self._row(next_state_key)
        next_value = 0.0 if done else float(np.max(np.where(_action_mask(next_valid_actions, self.n_actions), next_row, -1e12)))
        td_target = reward + self.discount * next_value
        current_row[action] += self.learning_rate * (td_target - current_row[action])
        self.steps += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "q_values": {key: value.copy() for key, value in self.q_values.items()},
            "steps": self.steps,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.q_values = {key: value.copy() for key, value in snapshot["q_values"].items()}
        self.steps = int(snapshot["steps"])


class SparseSARSAAgent:
    name = "sarsa"

    def __init__(
        self,
        n_actions: int,
        seed: int = 0,
        learning_rate: float = 0.11,
        discount: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 6000,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.q_values: dict[object, np.ndarray] = {}

    def _row(self, state_key) -> np.ndarray:
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_values[state_key]

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / float(max(1, self.epsilon_decay_steps)))
        return float(self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start))

    def act(self, state_key, training: bool = True, valid_actions=None) -> int:
        if training and self.rng.random() < self.epsilon():
            return _sample_masked_action(self.rng, valid_actions, self.n_actions)
        return _masked_argmax(self._row(state_key), valid_actions)

    def update(self, state_key, action: int, reward: float, next_state_key, next_action: int, done: bool) -> None:
        current_row = self._row(state_key)
        next_value = 0.0 if done else float(self._row(next_state_key)[next_action])
        td_target = reward + self.discount * next_value
        current_row[action] += self.learning_rate * (td_target - current_row[action])
        self.steps += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "q_values": {key: value.copy() for key, value in self.q_values.items()},
            "steps": self.steps,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.q_values = {key: value.copy() for key, value in snapshot["q_values"].items()}
        self.steps = int(snapshot["steps"])

DEFAULT_LAYOUT_PATH = Path(__file__).resolve().with_name("london_uber_layout.json")


@dataclass
class RideRequest:
    pickup_hub: int
    dropoff_hub: int
    status: int  # 0 waiting, 1 onboard, 2 delivered
    deadline_remaining: int
    wait_steps: int = 0
    ride_steps: int = 0
    was_late: bool = False
    is_placeholder: bool = False


class LondonUberDispatchEnv:
    """High-level rideshare dispatch environment on a real London, Ontario road graph."""

    STATUS_NAMES = {0: "waiting", 1: "onboard", 2: "delivered"}

    def __init__(self, seed: int = 0, layout_path: str | Path | None = None):
        self.layout_path = Path(layout_path) if layout_path is not None else DEFAULT_LAYOUT_PATH
        layout = json.loads(self.layout_path.read_text())

        self.city_name = layout["city_name"]
        self.width = int(layout["width"])
        self.height = int(layout["height"])
        self.road_cells = {tuple(cell) for cell in layout["road_cells"]}
        self.hubs = [tuple(item["cell"]) for item in layout["hubs"]]
        self.hub_names = [item["name"] for item in layout["hubs"]]
        self.hub_lookup = {name: idx for idx, name in enumerate(self.hub_names)}
        self.depots = [tuple(item["cell"]) for item in layout["depots"]]
        self.depot_names = [item["name"] for item in layout["depots"]]
        self.depot_lookup = {name: idx for idx, name in enumerate(self.depot_names)}
        self.base_traffic_penalties = {tuple(item["cell"]): float(item["penalty"]) for item in layout["traffic_penalties"]}
        self.phase_specs = layout["traffic_phases"]
        self.template_specs = layout["demand_templates"]
        self.n_phases = len(self.phase_specs)

        self.capacity = 2
        self.n_active_requests = 4
        self.current_request_count = self.n_active_requests
        self.current_phase_count = self.n_phases
        self.deadline_slack_bonus = 0
        self.rng = np.random.default_rng(seed)

        self.mobility_nodes = []
        self.mobility_names = []
        self.mobility_kind = []
        for name, cell in zip(self.depot_names, self.depots):
            self.mobility_nodes.append(cell)
            self.mobility_names.append(name)
            self.mobility_kind.append("depot")
        for name, cell in zip(self.hub_names, self.hubs):
            if cell not in self.mobility_nodes:
                self.mobility_nodes.append(cell)
                self.mobility_names.append(name)
                self.mobility_kind.append("hub")
        self.mobility_index = {cell: idx for idx, cell in enumerate(self.mobility_nodes)}

        self.distance_cache, self.parent_cache = self._build_bfs_cache()
        self.path_lookup = self._build_path_lookup()
        self.phase_traffic_penalties = self._build_phase_traffic_maps()
        self.templates = self._resolve_templates()
        self.max_trip = self._estimate_max_trip()
        self.max_deadline = max(56, min(96, self.max_trip + 30))
        self.max_steps = max(90, min(160, self.max_trip * 3 + 20))
        self.deadline_buckets = 5
        self.wait_buckets = 4
        self.status_values = 3
        self.n_actions = self.n_active_requests * 2
        self.obs_dim = (
            len(self.mobility_nodes)
            + (self.capacity + 1)
            + self.n_phases
            + self.n_actions * 2
            + self.n_active_requests * (len(self.hubs) + len(self.hubs) + self.status_values + self.deadline_buckets + self.wait_buckets + 5)
            + 6
        )
        self.reset()

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = cell
        neighbors = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            candidate = (x + dx, y + dy)
            if candidate in self.road_cells:
                neighbors.append(candidate)
        return neighbors

    def _bfs(self, start: tuple[int, int]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
        queue = deque([start])
        distance = {start: 0}
        parent = {start: None}
        while queue:
            cell = queue.popleft()
            for neighbor in self._neighbors(cell):
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[cell] + 1
                parent[neighbor] = cell
                queue.append(neighbor)
        return distance, parent

    def _build_bfs_cache(self):
        distances = {}
        parents = {}
        for start in self.mobility_nodes:
            distance, parent = self._bfs(start)
            distances[start] = distance
            parents[start] = parent
        return distances, parents

    def _path_from_parent(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        parent = self.parent_cache[start]
        path = [end]
        current = end
        while current != start:
            current = parent[current]
            if current is None:
                return [start]
            path.append(current)
        path.reverse()
        return path

    def _build_path_lookup(self) -> dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]:
        lookup = {}
        for start in self.mobility_nodes:
            for end in self.mobility_nodes:
                lookup[(start, end)] = self._path_from_parent(start, end)
        return lookup

    def road_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        if a in self.distance_cache:
            return int(self.distance_cache[a][b])
        distance_map, _ = self._bfs(a)
        return int(distance_map[b])

    def _build_phase_traffic_maps(self) -> list[dict[tuple[int, int], float]]:
        phase_maps = []
        for spec in self.phase_specs:
            penalties = dict(self.base_traffic_penalties)
            radius = int(spec["radius"])
            phase_penalty = float(spec["penalty"])
            for center_name in spec["centers"]:
                if center_name not in self.hub_lookup:
                    continue
                center_cell = self.hubs[self.hub_lookup[center_name]]
                distances = self.distance_cache.get(center_cell)
                if distances is None:
                    distance_map, _ = self._bfs(center_cell)
                    distances = distance_map
                for road_cell, distance in distances.items():
                    if distance > radius:
                        continue
                    penalty = phase_penalty * (1.0 if distance <= 1 else 0.7)
                    penalties[road_cell] = max(penalties.get(road_cell, 0.0), penalty)
            phase_maps.append(penalties)
        return phase_maps

    def _resolve_templates(self) -> list[dict]:
        templates = []
        for spec in self.template_specs:
            templates.append(
                {
                    "name": spec["name"],
                    "pickup_hubs": [self.hub_lookup[name] for name in spec["pickup_hubs"] if name in self.hub_lookup],
                    "dropoff_hubs": [self.hub_lookup[name] for name in spec["dropoff_hubs"] if name in self.hub_lookup],
                    "start_depots": [self.depot_lookup[name] for name in spec["start_hubs"] if name in self.depot_lookup],
                }
            )
        return templates

    def _estimate_max_trip(self) -> int:
        best = 0
        for template in self.templates:
            for depot_index in template["start_depots"]:
                depot_cell = self.depots[depot_index]
                for pickup_hub in template["pickup_hubs"]:
                    for dropoff_hub in template["dropoff_hubs"]:
                        if pickup_hub == dropoff_hub:
                            continue
                        trip = self.road_distance(depot_cell, self.hubs[pickup_hub]) + self.road_distance(self.hubs[pickup_hub], self.hubs[dropoff_hub])
                        best = max(best, trip)
        return best

    def set_request_count(self, request_count: int) -> None:
        self.current_request_count = max(1, min(self.n_active_requests, int(request_count)))

    def set_phase_count(self, phase_count: int) -> None:
        self.current_phase_count = max(1, min(self.n_phases, int(phase_count)))

    def set_deadline_slack_bonus(self, slack_bonus: int) -> None:
        self.deadline_slack_bonus = max(0, int(slack_bonus))

    def carrying_count(self) -> int:
        return sum(request.status == 1 for request in self.requests)

    def _position_anchor(self, position: tuple[int, int]) -> tuple[int, int]:
        if position in self.mobility_index:
            return position
        return min(self.mobility_nodes, key=lambda node: self.road_distance(position, node))

    def delivered_count(self) -> int:
        return sum(request.status == 2 and not request.is_placeholder for request in self.requests)

    def late_dropoff_count(self) -> int:
        return sum(request.was_late and not request.is_placeholder for request in self.requests)

    def _deadline_bucket(self, remaining: int) -> int:
        if remaining <= 0:
            return 0
        if remaining <= 8:
            return 1
        if remaining <= 16:
            return 2
        if remaining <= 28:
            return 3
        return 4

    def _wait_bucket(self, wait_steps: int) -> int:
        if wait_steps <= 4:
            return 0
        if wait_steps <= 10:
            return 1
        if wait_steps <= 18:
            return 2
        return 3

    def _sample_requests(self) -> list[RideRequest]:
        template = self.templates[self.phase_index]
        start_cell = tuple(self.driver_position)
        pickup_pool = template["pickup_hubs"]
        dropoff_pool = template["dropoff_hubs"]
        anchor_pickup = int(self.rng.choice(pickup_pool))
        anchor_dropoff = int(self.rng.choice(dropoff_pool))
        pickup_neighbors = sorted(
            pickup_pool,
            key=lambda hub_index: self.road_distance(self.hubs[anchor_pickup], self.hubs[hub_index]),
        )[: max(2, min(3, len(pickup_pool)))]
        dropoff_neighbors = sorted(
            dropoff_pool,
            key=lambda hub_index: self.road_distance(self.hubs[anchor_dropoff], self.hubs[hub_index]),
        )[: max(2, min(3, len(dropoff_pool)))]

        requests = []
        for request_index in range(self.current_request_count):
            pickup_candidates = pickup_neighbors if request_index > 0 and self.rng.random() < 0.7 else pickup_pool
            dropoff_candidates = dropoff_neighbors if request_index > 0 and self.rng.random() < 0.7 else dropoff_pool
            pickup_hub = anchor_pickup if request_index == 0 else int(self.rng.choice(pickup_candidates))
            dropoff_hub = anchor_dropoff if request_index == 0 else int(self.rng.choice(dropoff_candidates))
            if pickup_hub == dropoff_hub and len(dropoff_pool) > 1:
                alternatives = [hub for hub in dropoff_pool if hub != pickup_hub]
                dropoff_hub = int(self.rng.choice(alternatives))
            direct_trip = self.road_distance(self.hubs[pickup_hub], self.hubs[dropoff_hub])
            pickup_trip = self.road_distance(start_cell, self.hubs[pickup_hub])
            deadline = int(min(self.max_deadline, max(24, pickup_trip + direct_trip + 12 + self.deadline_slack_bonus)))
            requests.append(
                RideRequest(
                    pickup_hub=pickup_hub,
                    dropoff_hub=dropoff_hub,
                    status=0,
                    deadline_remaining=deadline,
                )
            )
        while len(requests) < self.n_active_requests:
            requests.append(
                RideRequest(
                    pickup_hub=0,
                    dropoff_hub=0,
                    status=2,
                    deadline_remaining=self.max_deadline,
                    is_placeholder=True,
                )
            )
        return requests

    def reset(
        self,
        seed: int | None = None,
        request_count: int | None = None,
        phase_count: int | None = None,
        deadline_slack_bonus: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        if request_count is not None:
            self.set_request_count(request_count)
        if phase_count is not None:
            self.set_phase_count(phase_count)
        if deadline_slack_bonus is not None:
            self.set_deadline_slack_bonus(deadline_slack_bonus)

        self.phase_index = int(self.rng.integers(self.current_phase_count))
        template = self.templates[self.phase_index]
        depot_index = int(self.rng.choice(template["start_depots"]))
        self.driver_position = list(self.depots[depot_index])
        self.requests = self._sample_requests()
        self.steps_taken = 0
        self.done = False
        self.success = False
        self.peak_occupancy = 0
        self.total_path = [tuple(self.driver_position)]
        return self.observation(), self.info()

    def _action_target(self, action: int) -> tuple[int, tuple[int, int]]:
        request_index = action % self.n_active_requests
        request = self.requests[request_index]
        if action < self.n_active_requests:
            return request_index, self.hubs[request.pickup_hub]
        return request_index, self.hubs[request.dropoff_hub]

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        carrying = self.carrying_count()
        for request_index, request in enumerate(self.requests):
            if request.is_placeholder:
                continue
            mask[request_index] = request.status == 0 and carrying < self.capacity
            mask[self.n_active_requests + request_index] = request.status == 1
        return mask

    def state_key(self):
        position = self._position_anchor(tuple(self.driver_position))
        request_signature = tuple(
            (
                request.pickup_hub,
                request.dropoff_hub,
                request.status,
                self._deadline_bucket(request.deadline_remaining),
                self._wait_bucket(request.wait_steps),
            )
            for request in self.requests
        )
        return (position, self.phase_index, self.carrying_count(), request_signature)

    def observation(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        cursor = 0
        raw_position = tuple(self.driver_position)
        position = self._position_anchor(raw_position)
        obs[cursor + self.mobility_index[position]] = 1.0
        cursor += len(self.mobility_nodes)
        obs[cursor + self.carrying_count()] = 1.0
        cursor += self.capacity + 1
        obs[cursor + self.phase_index] = 1.0
        cursor += self.n_phases

        for action in range(self.n_actions):
            request_index, target = self._action_target(action)
            legal = self.valid_action_mask()[action]
            obs[cursor] = 1.0 if legal else 0.0
            obs[cursor + 1] = 1.0 if not legal else min(1.0, self.road_distance(position, target) / max(1.0, float(self.max_trip)))
            cursor += 2

        for request in self.requests:
            obs[cursor + request.pickup_hub] = 1.0
            cursor += len(self.hubs)
            obs[cursor + request.dropoff_hub] = 1.0
            cursor += len(self.hubs)
            obs[cursor + request.status] = 1.0
            cursor += self.status_values
            obs[cursor + self._deadline_bucket(request.deadline_remaining)] = 1.0
            cursor += self.deadline_buckets
            obs[cursor + self._wait_bucket(request.wait_steps)] = 1.0
            cursor += self.wait_buckets
            current_target = self.hubs[request.pickup_hub] if request.status == 0 else self.hubs[request.dropoff_hub]
            obs[cursor] = min(1.0, self.road_distance(raw_position, current_target) / max(1.0, float(self.max_trip))) if request.status != 2 else 0.0
            obs[cursor + 1] = request.deadline_remaining / max(1.0, float(self.max_deadline))
            obs[cursor + 2] = request.wait_steps / max(1.0, float(self.max_steps))
            obs[cursor + 3] = request.ride_steps / max(1.0, float(self.max_steps))
            obs[cursor + 4] = 1.0 if request.was_late else 0.0
            cursor += 5

        obs[cursor] = raw_position[0] / max(1.0, float(self.width - 1))
        obs[cursor + 1] = raw_position[1] / max(1.0, float(self.height - 1))
        obs[cursor + 2] = self.delivered_count() / max(1.0, float(self.current_request_count))
        obs[cursor + 3] = self.late_dropoff_count() / max(1.0, float(self.current_request_count))
        obs[cursor + 4] = self.peak_occupancy / max(1.0, float(self.capacity))
        obs[cursor + 5] = self.steps_taken / max(1.0, float(self.max_steps))
        return obs

    def info(self) -> dict:
        return {
            "city_name": self.city_name,
            "state_key": self.state_key(),
            "driver_position": tuple(self.driver_position),
            "steps": int(self.steps_taken),
            "max_steps": int(self.max_steps),
            "carrying_count": int(self.carrying_count()),
            "delivered_count": int(self.delivered_count()),
            "required_deliveries": int(self.current_request_count),
            "late_dropoffs": int(self.late_dropoff_count()),
            "peak_occupancy": int(self.peak_occupancy),
            "pooled_episode": bool(self.peak_occupancy >= 2),
            "traffic_cells": set(self.phase_traffic_penalties[self.phase_index].keys()),
            "phase_index": int(self.phase_index),
            "phase_name": self.phase_specs[self.phase_index]["name"],
            "scenario_name": self.templates[self.phase_index]["name"],
            "valid_actions": self.valid_action_mask(),
            "requests": [
                {
                    "pickup_hub": int(request.pickup_hub),
                    "pickup_name": self.hub_names[request.pickup_hub],
                    "pickup_cell": self.hubs[request.pickup_hub],
                    "dropoff_hub": int(request.dropoff_hub),
                    "dropoff_name": self.hub_names[request.dropoff_hub],
                    "dropoff_cell": self.hubs[request.dropoff_hub],
                    "status": int(request.status),
                    "status_name": self.STATUS_NAMES[request.status],
                    "deadline_remaining": int(request.deadline_remaining),
                    "wait_steps": int(request.wait_steps),
                    "ride_steps": int(request.ride_steps),
                    "was_late": bool(request.was_late),
                    "is_placeholder": bool(request.is_placeholder),
                }
                for request in self.requests
            ],
        }

    def _advance_time_along_path(self, path: list[tuple[int, int]]) -> float:
        reward = 0.0
        for cell in path[1:]:
            self.driver_position = [cell[0], cell[1]]
            self.total_path.append(cell)
            self.steps_taken += 1
            reward -= 0.18
            reward -= 0.04 * max(0, self.current_request_count - self.delivered_count())
            reward -= self.phase_traffic_penalties[self.phase_index].get(cell, 0.0)
            reward += 0.12 * max(0, self.carrying_count() - 1)
            for request in self.requests:
                if request.status == 2 or request.is_placeholder:
                    continue
                request.deadline_remaining = max(0, request.deadline_remaining - 1)
                if request.status == 0:
                    request.wait_steps += 1
                else:
                    request.ride_steps += 1
            if self.steps_taken >= self.max_steps:
                self.done = True
                reward -= 12.0
                break
        return reward

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        if not self.valid_action_mask()[action]:
            self.steps_taken += 1
            reward = -8.0
            for request in self.requests:
                if request.status == 2 or request.is_placeholder:
                    continue
                request.deadline_remaining = max(0, request.deadline_remaining - 1)
                if request.status == 0:
                    request.wait_steps += 1
                else:
                    request.ride_steps += 1
            if self.steps_taken >= self.max_steps:
                self.done = True
                reward -= 12.0
            return self.observation(), reward, self.done, self.info()

        request_index, target = self._action_target(action)
        request = self.requests[request_index]
        current_position = tuple(self.driver_position)
        path = self.path_lookup[(current_position, target)]
        reward = self._advance_time_along_path(path)

        if not self.done:
            if action < self.n_active_requests:
                request.status = 1
                reward += 10.0
                if self.carrying_count() >= 2:
                    reward += 4.0
            else:
                request.status = 2
                request.was_late = request.deadline_remaining <= 0
                reward += 28.0 if not request.was_late else 18.0

        self.peak_occupancy = max(self.peak_occupancy, self.carrying_count())
        if not self.done and self.delivered_count() == self.current_request_count:
            self.done = True
            self.success = True
            reward += 22.0
            if self.late_dropoff_count() == 0:
                reward += 10.0
            if self.peak_occupancy >= 2:
                reward += 8.0

        return self.observation(), float(reward), bool(self.done), self.info()

    def action_name(self, action: int) -> str:
        request_index = action % self.n_active_requests
        return f"{'pickup' if action < self.n_active_requests else 'dropoff'} rider {request_index + 1}"


ALGORITHMS = ("q_learning", "sarsa", "dqn", "ppo")
HEURISTICS = ("nearest_stop", "earliest_deadline")
DEFAULT_AGENT_CONFIGS = {
    "q_learning": {"learning_rate": 0.18, "discount": 0.99, "epsilon_decay_steps": 14000},
    "sarsa": {"learning_rate": 0.14, "discount": 0.985, "epsilon_decay_steps": 12000},
    "dqn": {
        "hidden_dim": 128,
        "learning_rate": 0.0011,
        "discount": 0.99,
        "epsilon_decay_steps": 18000,
        "warmup_steps": 400,
        "target_sync_steps": 180,
        "batch_size": 96,
        "buffer_size": 16000,
    },
    "ppo": {
        "hidden_dim": 128,
        "policy_lr": 0.0010,
        "value_lr": 0.0020,
        "discount": 0.99,
        "rollout_episodes": 8,
        "update_epochs": 8,
        "minibatch_size": 96,
    },
}


def _target_distance(env: LondonUberDispatchEnv, info: dict, action: int) -> int:
    _, target = env._action_target(action)
    return env.road_distance(tuple(info["driver_position"]), target)


def heuristic_action(policy_name: str, env: LondonUberDispatchEnv, info: dict) -> int:
    valid_actions = np.flatnonzero(np.asarray(info["valid_actions"], dtype=bool))
    if len(valid_actions) == 0:
        return 0

    if policy_name == "nearest_stop":
        ranked = sorted(valid_actions, key=lambda action: (_target_distance(env, info, int(action)), int(action)))
        return int(ranked[0])

    if policy_name == "earliest_deadline":
        def rank(action: int) -> tuple[float, float, int]:
            request_index = int(action) % env.n_active_requests
            request = info["requests"][request_index]
            dropoff_bias = 0 if int(action) >= env.n_active_requests else 1
            return (
                request["deadline_remaining"],
                dropoff_bias,
                _target_distance(env, info, int(action)),
            )

        ranked = sorted(valid_actions, key=rank)
        return int(ranked[0])

    raise ValueError(f"Unknown heuristic policy: {policy_name}")


def make_agent(name: str, env: LondonUberDispatchEnv, seed: int, config: dict | None = None):
    config = {**DEFAULT_AGENT_CONFIGS[name], **(config or {})}
    if name == "q_learning":
        return SparseQLearningAgent(
            n_actions=env.n_actions,
            seed=seed,
            learning_rate=config["learning_rate"],
            discount=config["discount"],
            epsilon_decay_steps=config["epsilon_decay_steps"],
        )
    if name == "sarsa":
        return SparseSARSAAgent(
            n_actions=env.n_actions,
            seed=seed,
            learning_rate=config["learning_rate"],
            discount=config["discount"],
            epsilon_decay_steps=config["epsilon_decay_steps"],
        )
    if name == "dqn":
        return DQNAgent(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            seed=seed,
            hidden_dim=config["hidden_dim"],
            learning_rate=config["learning_rate"],
            discount=config["discount"],
            epsilon_decay_steps=config["epsilon_decay_steps"],
            warmup_steps=config["warmup_steps"],
            target_sync_steps=config["target_sync_steps"],
            batch_size=config["batch_size"],
            buffer_size=config["buffer_size"],
        )
    if name == "ppo":
        return PPOAgent(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            seed=seed,
            hidden_dim=config["hidden_dim"],
            policy_lr=config["policy_lr"],
            value_lr=config["value_lr"],
            discount=config["discount"],
            rollout_episodes=config["rollout_episodes"],
            update_epochs=config["update_epochs"],
            minibatch_size=config["minibatch_size"],
        )
    raise ValueError(f"Unknown algorithm: {name}")


def greedy_action(agent, obs: np.ndarray, state_key, valid_actions) -> int:
    if isinstance(agent, (SparseQLearningAgent, SparseSARSAAgent)):
        return agent.act(state_key, training=False, valid_actions=valid_actions)
    if isinstance(agent, DQNAgent):
        return agent.act(obs, training=False, valid_actions=valid_actions)
    if isinstance(agent, PPOAgent):
        action, _, _ = agent.act(obs, training=False, valid_actions=valid_actions)
        return action
    raise TypeError(f"Unsupported agent type: {type(agent)!r}")


def train_agent(
    algorithm_name: str,
    episodes: int,
    seed: int,
    config: dict | None = None,
    layout_path: str | Path | None = None,
    *,
    use_curriculum: bool = True,
    use_action_mask: bool = True,
) -> dict:
    env = LondonUberDispatchEnv(seed=seed, layout_path=layout_path)
    agent = make_agent(algorithm_name, env, seed, config=config)
    rewards, delivered, pooled, late = [], [], [], []

    for episode in range(1, episodes + 1):
        progress = episode / float(max(1, episodes))
        if use_curriculum and progress < 0.25:
            request_count, phase_count, slack_bonus = 2, 1, 18
        elif use_curriculum and progress < 0.6:
            request_count, phase_count, slack_bonus = env.n_active_requests, 2, 14
        else:
            request_count, phase_count, slack_bonus = env.n_active_requests, env.n_phases, 10

        obs, info = env.reset(request_count=request_count, phase_count=phase_count, deadline_slack_bonus=slack_bonus)
        state_key = info["state_key"]
        valid_actions = info["valid_actions"] if use_action_mask else None
        episode_reward = 0.0
        done = False

        if isinstance(agent, SparseSARSAAgent):
            action = agent.act(state_key, training=True, valid_actions=valid_actions)
        if isinstance(agent, PPOAgent):
            ep_obs, ep_actions, ep_rewards, ep_log_probs, ep_values, ep_dones, ep_masks = [], [], [], [], [], [], []

        while not done:
            if isinstance(agent, SparseQLearningAgent):
                action = agent.act(state_key, training=True, valid_actions=valid_actions)
                next_obs, reward, done, next_info = env.step(action)
                next_state_key = next_info["state_key"]
                next_valid_actions = next_info["valid_actions"] if use_action_mask else None
                agent.update(state_key, action, reward, next_state_key, done, next_valid_actions=next_valid_actions)
                obs, state_key, valid_actions = next_obs, next_state_key, next_valid_actions
            elif isinstance(agent, SparseSARSAAgent):
                next_obs, reward, done, next_info = env.step(action)
                next_state_key = next_info["state_key"]
                next_valid_actions = next_info["valid_actions"] if use_action_mask else None
                next_action = 0 if done else agent.act(next_state_key, training=True, valid_actions=next_valid_actions)
                agent.update(state_key, action, reward, next_state_key, next_action, done)
                obs, state_key, action, valid_actions = next_obs, next_state_key, next_action, next_valid_actions
            elif isinstance(agent, DQNAgent):
                action = agent.act(obs, training=True, valid_actions=valid_actions)
                next_obs, reward, done, next_info = env.step(action)
                next_valid_actions = next_info["valid_actions"] if use_action_mask else None
                agent.store(obs, action, reward, next_obs, done, next_action_mask=next_valid_actions)
                agent.update()
                obs, state_key, valid_actions = next_obs, next_info["state_key"], next_valid_actions
            elif isinstance(agent, PPOAgent):
                action, log_prob, value = agent.act(obs, training=True, valid_actions=valid_actions)
                next_obs, reward, done, next_info = env.step(action)
                ep_obs.append(obs.copy())
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_log_probs.append(log_prob)
                ep_values.append(value)
                ep_dones.append(done)
                ep_masks.append(np.asarray(_action_mask(valid_actions, env.n_actions), dtype=bool))
                obs, state_key, valid_actions = next_obs, next_info["state_key"], (next_info["valid_actions"] if use_action_mask else None)
            else:
                raise TypeError(f"Unsupported agent type: {type(agent)!r}")
            episode_reward += reward

        if isinstance(agent, PPOAgent):
            agent.remember_episode(ep_obs, ep_actions, ep_rewards, ep_log_probs, ep_values, ep_dones, ep_masks)
            if agent.should_update():
                agent.update()

        rewards.append(float(episode_reward))
        delivered.append(float(env.delivered_count()))
        pooled.append(1.0 if env.peak_occupancy >= 2 else 0.0)
        late.append(float(env.late_dropoff_count()))

    if isinstance(agent, PPOAgent) and agent.trajectory_buffer:
        agent.update()

    return {
        "agent": agent,
        "training_rewards": np.array(rewards, dtype=np.float32),
        "training_delivered": np.array(delivered, dtype=np.float32),
        "training_pooled": np.array(pooled, dtype=np.float32),
        "training_late": np.array(late, dtype=np.float32),
        "config": config or DEFAULT_AGENT_CONFIGS[algorithm_name],
        "use_curriculum": use_curriculum,
        "use_action_mask": use_action_mask,
    }


def _summarize_episode_metrics(
    rewards: list[float],
    successes: list[float],
    steps: list[float],
    deliveries: list[float],
    pooled: list[float],
    late_dropoffs: list[float],
    peak_occupancy: list[float],
    on_time_rates: list[float],
) -> dict:
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps)),
        "mean_deliveries": float(np.mean(deliveries)),
        "pooled_rate": float(np.mean(pooled)),
        "mean_late_dropoffs": float(np.mean(late_dropoffs)),
        "mean_peak_occupancy": float(np.mean(peak_occupancy)),
        "on_time_rate": float(np.mean(on_time_rates)),
    }


def evaluate_agent(agent, eval_episodes: int, seed: int, layout_path: str | Path | None = None, *, use_action_mask: bool = True) -> dict:
    env = LondonUberDispatchEnv(seed=seed, layout_path=layout_path)
    rewards, successes, steps, deliveries, pooled = [], [], [], [], []
    late_dropoffs, peak_occupancy, on_time_rates = [], [], []

    for episode in range(eval_episodes):
        obs, info = env.reset(seed=seed * 1000 + episode, request_count=env.n_active_requests, phase_count=env.n_phases, deadline_slack_bonus=10)
        state_key = info["state_key"]
        valid_actions = info["valid_actions"] if use_action_mask else None
        done = False
        episode_reward = 0.0
        while not done:
            action = greedy_action(agent, obs, state_key, valid_actions)
            obs, reward, done, info = env.step(action)
            state_key = info["state_key"]
            valid_actions = info["valid_actions"] if use_action_mask else None
            episode_reward += reward
        rewards.append(episode_reward)
        successes.append(1.0 if env.success else 0.0)
        steps.append(info["steps"])
        deliveries.append(info["delivered_count"])
        pooled.append(1.0 if env.peak_occupancy >= 2 else 0.0)
        late_dropoffs.append(info["late_dropoffs"])
        peak_occupancy.append(info["peak_occupancy"])
        on_time = 0.0 if info["delivered_count"] == 0 else max(0.0, (info["delivered_count"] - info["late_dropoffs"]) / float(info["delivered_count"]))
        on_time_rates.append(on_time)

    return _summarize_episode_metrics(rewards, successes, steps, deliveries, pooled, late_dropoffs, peak_occupancy, on_time_rates)


def evaluate_heuristic(policy_name: str, eval_episodes: int, seed: int, layout_path: str | Path | None = None) -> dict:
    env = LondonUberDispatchEnv(seed=seed, layout_path=layout_path)
    rewards, successes, steps, deliveries, pooled = [], [], [], [], []
    late_dropoffs, peak_occupancy, on_time_rates = [], [], []

    for episode in range(eval_episodes):
        _, info = env.reset(seed=seed * 1000 + episode, request_count=env.n_active_requests, phase_count=env.n_phases, deadline_slack_bonus=10)
        done = False
        episode_reward = 0.0
        while not done:
            action = heuristic_action(policy_name, env, info)
            _, reward, done, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        successes.append(1.0 if env.success else 0.0)
        steps.append(info["steps"])
        deliveries.append(info["delivered_count"])
        pooled.append(1.0 if env.peak_occupancy >= 2 else 0.0)
        late_dropoffs.append(info["late_dropoffs"])
        peak_occupancy.append(info["peak_occupancy"])
        on_time = 0.0 if info["delivered_count"] == 0 else max(0.0, (info["delivered_count"] - info["late_dropoffs"]) / float(info["delivered_count"]))
        on_time_rates.append(on_time)

    return _summarize_episode_metrics(rewards, successes, steps, deliveries, pooled, late_dropoffs, peak_occupancy, on_time_rates)


def rolling_mean(values: np.ndarray, window: int = 120) -> np.ndarray:
    if len(values) == 0:
        return values
    window = min(window, len(values))
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    rolling = (cumsum[window:] - cumsum[:-window]) / float(window)
    prefix = np.full(window - 1, rolling[0], dtype=np.float32)
    return np.concatenate([prefix, rolling.astype(np.float32)])


def aggregate_results(run_results: dict[str, list[dict]], eval_episodes: int, layout_path: str | Path | None = None) -> dict:
    aggregated = {}
    for algorithm_name, seed_runs in run_results.items():
        reward_curves = np.stack([run["training_rewards"] for run in seed_runs], axis=0)
        delivered_curves = np.stack([run["training_delivered"] for run in seed_runs], axis=0)
        pooled_curves = np.stack([run["training_pooled"] for run in seed_runs], axis=0)
        eval_metrics = []
        for seed_index, run in enumerate(seed_runs):
            eval_metrics.append(
                evaluate_agent(
                    run["agent"],
                    eval_episodes=eval_episodes,
                    seed=7400 + seed_index,
                    layout_path=layout_path,
                    use_action_mask=run.get("use_action_mask", True),
                )
            )
        aggregated[algorithm_name] = {
            "seed_runs": seed_runs,
            "mean_training_rewards": reward_curves.mean(axis=0),
            "mean_training_delivered": delivered_curves.mean(axis=0),
            "mean_training_pooled": pooled_curves.mean(axis=0),
            "evaluation": {key: float(np.mean([metric[key] for metric in eval_metrics])) for key in eval_metrics[0]},
            "evaluation_std": {key: float(np.std([metric[key] for metric in eval_metrics])) for key in eval_metrics[0]},
            "config": seed_runs[0]["config"],
        }
    return aggregated


def evaluate_heuristics(eval_episodes: int, seeds: int, layout_path: str | Path | None = None) -> dict:
    results = {}
    for policy_name in HEURISTICS:
        per_seed = [evaluate_heuristic(policy_name, eval_episodes=eval_episodes, seed=9100 + seed, layout_path=layout_path) for seed in range(seeds)]
        results[policy_name] = {
            "evaluation": {key: float(np.mean([metric[key] for metric in per_seed])) for key in per_seed[0]},
            "evaluation_std": {key: float(np.std([metric[key] for metric in per_seed])) for key in per_seed[0]},
        }
    return results


def run_dqn_ablations(episodes: int, seeds: int, eval_episodes: int, layout_path: str | Path | None = None) -> dict:
    variants = {
        "dqn_full": {"use_curriculum": True, "use_action_mask": True},
        "dqn_no_curriculum": {"use_curriculum": False, "use_action_mask": True},
        "dqn_no_mask": {"use_curriculum": True, "use_action_mask": False},
    }
    results = {}
    for variant_name, variant in variants.items():
        seed_runs = []
        for seed in range(seeds):
            seed_runs.append(
                train_agent(
                    "dqn",
                    episodes=episodes,
                    seed=100 + seed,
                    config=DEFAULT_AGENT_CONFIGS["dqn"],
                    layout_path=layout_path,
                    use_curriculum=variant["use_curriculum"],
                    use_action_mask=variant["use_action_mask"],
                )
            )
        aggregated = aggregate_results({variant_name: seed_runs}, eval_episodes=eval_episodes, layout_path=layout_path)[variant_name]
        results[variant_name] = aggregated
    return results


def plot_results(results: dict, output_dir: Path, heuristic_results: dict | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    for algorithm_name, result in results.items():
        label = algorithm_name.replace("_", " ").title()
        axes[0].plot(rolling_mean(result["mean_training_rewards"]), label=label, linewidth=2)
        axes[1].plot(rolling_mean(result["mean_training_delivered"]), label=label, linewidth=2)
        axes[2].plot(rolling_mean(result["mean_training_pooled"]), label=label, linewidth=2)
    axes[0].set_title("Rolling Training Reward"); axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_title("Rolling Riders Delivered"); axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Delivered Riders"); axes[1].grid(alpha=0.3)
    axes[2].set_title("Rolling Pooling Rate"); axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Episodes using 2 riders"); axes[2].grid(alpha=0.3)
    comparison = {**results}
    if heuristic_results:
        comparison.update(heuristic_results)
    labels = [name.replace("_", " ").title() for name in comparison]
    x = np.arange(len(labels))
    on_time = [comparison[name]["evaluation"]["on_time_rate"] * 100.0 for name in comparison]
    deliveries = [comparison[name]["evaluation"]["mean_deliveries"] for name in comparison]
    pooling = [comparison[name]["evaluation"]["pooled_rate"] * 100.0 for name in comparison]
    width = 0.26
    axes[3].bar(x - width, on_time, width=width, label="On-Time Riders (%)", color="#2563eb")
    axes[3].bar(x, deliveries, width=width, label="Mean Riders", color="#f59e0b")
    axes[3].bar(x + width, pooling, width=width, label="Pooling Rate (%)", color="#10b981")
    axes[3].set_title("Evaluation Comparison"); axes[3].set_xticks(x); axes[3].set_xticklabels(labels, rotation=10); axes[3].grid(alpha=0.25, axis="y"); axes[3].legend()
    fig.tight_layout(); fig.savefig(output_dir / "results_summary.png", dpi=180); plt.close(fig)


def save_summary(results: dict, output_dir: Path, city_name: str, heuristic_results: dict | None = None, ablation_results: dict | None = None) -> None:
    payload = {}
    lines = [f"{city_name} Uber dispatch RL comparison", "=" * (len(city_name) + 34), ""]
    for algorithm_name, result in results.items():
        evaluation = result["evaluation"]
        payload[algorithm_name] = {
            "evaluation": evaluation,
            "evaluation_std": result.get("evaluation_std", {}),
            "mean_training_reward": float(np.mean(result["mean_training_rewards"])),
            "mean_training_delivered": float(np.mean(result["mean_training_delivered"])),
            "mean_training_pooled": float(np.mean(result["mean_training_pooled"])),
            "config": result["config"],
        }
        lines.extend([
            algorithm_name.replace("_", " ").title(),
            f"  mean evaluation reward: {evaluation['mean_reward']:.2f}",
            f"  full success rate: {evaluation['success_rate'] * 100.0:.1f}%",
            f"  on-time rider rate: {evaluation['on_time_rate'] * 100.0:.1f}%",
            f"  mean riders delivered: {evaluation['mean_deliveries']:.2f}",
            f"  mean late dropoffs: {evaluation['mean_late_dropoffs']:.2f}",
            f"  pooling rate: {evaluation['pooled_rate'] * 100.0:.1f}%",
            f"  mean peak occupancy: {evaluation['mean_peak_occupancy']:.2f}",
            f"  mean evaluation steps: {evaluation['mean_steps']:.2f}",
            "",
        ])
    if heuristic_results:
        lines.extend(["Heuristic baselines", "-------------------", ""])
        payload["heuristics"] = {}
        for name, result in heuristic_results.items():
            evaluation = result["evaluation"]
            payload["heuristics"][name] = result
            lines.extend([
                name.replace("_", " ").title(),
                f"  mean evaluation reward: {evaluation['mean_reward']:.2f}",
                f"  full success rate: {evaluation['success_rate'] * 100.0:.1f}%",
                f"  on-time rider rate: {evaluation['on_time_rate'] * 100.0:.1f}%",
                f"  mean late dropoffs: {evaluation['mean_late_dropoffs']:.2f}",
                f"  pooling rate: {evaluation['pooled_rate'] * 100.0:.1f}%",
                f"  mean evaluation steps: {evaluation['mean_steps']:.2f}",
                "",
            ])
    if ablation_results:
        lines.extend(["DQN ablations", "-------------", ""])
        payload["ablations"] = {}
        for name, result in ablation_results.items():
            evaluation = result["evaluation"]
            payload["ablations"][name] = {
                "evaluation": evaluation,
                "evaluation_std": result.get("evaluation_std", {}),
                "config": result["config"],
            }
            lines.extend([
                name.replace("_", " ").title(),
                f"  on-time rider rate: {evaluation['on_time_rate'] * 100.0:.1f}%",
                f"  mean late dropoffs: {evaluation['mean_late_dropoffs']:.2f}",
                f"  pooling rate: {evaluation['pooled_rate'] * 100.0:.1f}%",
                f"  mean evaluation steps: {evaluation['mean_steps']:.2f}",
                "",
            ])
    (output_dir / "summary.txt").write_text("\n".join(lines))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents on the London Uber dispatch benchmark.")
    parser.add_argument("--episodes", type=int, default=900)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="final_results")
    parser.add_argument("--layout-path", default=str(DEFAULT_LAYOUT_PATH))
    parser.add_argument("--include-heuristics", action="store_true")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--ablation-episodes", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_env = LondonUberDispatchEnv(seed=0, layout_path=args.layout_path)
    run_results = {name: [] for name in ALGORITHMS}
    for algorithm_name in ALGORITHMS:
        for seed in range(args.seeds):
            run_results[algorithm_name].append(train_agent(algorithm_name, args.episodes, seed, config=DEFAULT_AGENT_CONFIGS[algorithm_name], layout_path=args.layout_path))
    aggregated = aggregate_results(run_results, eval_episodes=args.eval_episodes, layout_path=args.layout_path)
    heuristic_results = evaluate_heuristics(args.eval_episodes, args.seeds, layout_path=args.layout_path) if args.include_heuristics else None
    ablation_results = run_dqn_ablations(args.ablation_episodes, args.seeds, args.eval_episodes, layout_path=args.layout_path) if args.run_ablations else None
    plot_results(aggregated, output_dir, heuristic_results=heuristic_results)
    save_summary(aggregated, output_dir, city_name=sample_env.city_name, heuristic_results=heuristic_results, ablation_results=ablation_results)
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
