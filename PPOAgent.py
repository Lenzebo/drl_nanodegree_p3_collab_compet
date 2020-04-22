import random
import torch
from torch import optim
import torch.nn as nn

import numpy as np

import BatchSelection

from HyperParameter import HyperParameter


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, device, policy,
                 hyperparameter: HyperParameter = HyperParameter()):
        """Initialize an Agent object.

        Params
        ======
            num_agents (int): number of parallel acting agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            device (str): device identifier used for the network
            policy: Policy
            hyperparameter: Parameters for training the network
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.hyperparameter = hyperparameter

        self.optimizier = optim.Adam(policy.parameters(), self.hyperparameter.learning_rate,
                                     eps=self.hyperparameter.adam_epsilon)

        self.policy = policy

    def perform_rollout(self, environment, brain_name):
        """ This function generates a rollout in the environment with the current policy

        :param environment: an environmnet adhering to the UnityEnvironment interface
        :param brain_name: name of the agent within the environment
        :return: a rollout (tuple of (states, values, actions, log_probs, rewards, terminals) and the value to go from the last state of the rollout
        """
        rollout = []
        env_info = environment.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        for _ in range(self.hyperparameter.rollout_length):
            actions, log_probs, _, values = self.policy(states)
            env_info = environment.step(actions.cpu().detach().numpy())[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            terminals = np.array([1 if t else 0 for t in env_info.local_done], dtype=np.float32)

            rollout.append([states, values.cpu().detach(), actions.cpu().detach(), log_probs.cpu().detach(), rewards,
                            1 - terminals])

            states = next_states

        # approximate the value from rollout_length -> Infty
        pending_value = self.policy(states)[-1]
        rollout.append([states, pending_value.cpu(), None, None, None, None])
        return rollout, pending_value

    def calculate_advantages(self, rollout, pending_value):
        """ This function calculates the advantages of a rollout,
        by backtracking rewards from the reached state of the rollout back to the beginning state

        :param rollout: tuple of (states, values, actions, log_probs, rewards, terminals)
        :param pending_value: value to go from the last state of the rollout
        :return: states, actions, log_probs, returns and advantages of the rollout as torch Tensors
        """
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.Tensor(np.zeros((self.num_agents, 1)))
        returns = pending_value.cpu().detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = torch.Tensor(terminals).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]
            discounted_returns = self.hyperparameter.discount_rate * terminals * returns
            returns = rewards + discounted_returns

            # Formula (12) in PPO paper
            td_error = rewards + self.hyperparameter.discount_rate * terminals * next_value.detach() - value.detach()
            # Formula (11) in PPO paper
            advantages = advantages * self.hyperparameter.lambda_ * self.hyperparameter.discount_rate * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0),
                                                                  zip(*processed_rollout))
        # normalize the advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        return states, actions, log_probs_old, returns, advantages

    def train_step(self, environment, brain_name):
        """
        Perform one training step according to https://arxiv.org/abs/1707.06347

        :param environment:
        :param brain_name:
        :return:
        """

        rollout, pending_value = self.perform_rollout(environment, brain_name)

        if not rollout:
            return
        states, actions, log_probs_old, returns, advantages = self.calculate_advantages(rollout, pending_value)

        if states.size(0) < self.hyperparameter.mini_batch_number:
            return

        batcher = BatchSelection.Batcher(states.size(0) // self.hyperparameter.mini_batch_number,
                                         [np.arange(states.size(0))])
        for _ in range(self.hyperparameter.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]

                if batch_indices.size:
                    batch_indices = torch.Tensor(batch_indices).long()

                    self.train_batch(states[batch_indices], actions[batch_indices], log_probs_old[batch_indices],
                                     returns[batch_indices], advantages[batch_indices])

    def train_batch(self, sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages):
        """
        Update the policy network with one set of sampled values (minibatch)
        :param sampled_states:
        :param sampled_actions:
        :param sampled_log_probs_old:
        :param sampled_returns:
        :param sampled_advantages:
        :return:
        """

        _, log_probs, entropy_loss, values = self.policy(sampled_states, sampled_actions)
        values = values.cpu()
        log_probs = log_probs.cpu()
        ratio = (log_probs - sampled_log_probs_old).exp()
        obj = ratio * sampled_advantages
        obj_clipped = ratio.clamp(1.0 - self.hyperparameter.ppo_clip,
                                  1.0 + self.hyperparameter.ppo_clip) * sampled_advantages
        policy_loss = -torch.min(obj, obj_clipped).mean(
            0) - self.hyperparameter.entropy_coefficent * entropy_loss.mean()
        value_loss = 0.5 * (sampled_returns - values).pow(2).mean()
        self.optimizier.zero_grad()
        (policy_loss + value_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyperparameter.gradient_clip)
        self.optimizier.step()

    def save(self, filename="model/model.pth"):
        """
        Save policy weigths to file
        :param filename:
        :return:
        """
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename="model/model.pth"):
        """
        Loads policy weights from file
        :param filename:
        :return:
        """
        self.policy.load_state_dict(torch.load(filename))

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        self.policy.eval()
        with torch.no_grad():
            actions, _, _, _ = self.policy(state)
        self.policy.train()

        action = actions.cpu().data.numpy()

        return action
