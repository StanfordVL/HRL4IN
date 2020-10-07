#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import gym

from hrl4in.utils.distributions import CategoricalNet, DiagGaussianNet, MultiCategoricalNet
from hrl4in.utils.networks import Net
from collections import OrderedDict

EPS = 1e-6
OLD_NETWORK = False

class Policy(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 use_camera_masks=False,
                 split_network=False, 
                 camera_masks_dim=3,
                 hidden_size=512,
                 cnn_layers_params=None,
                 initial_stddev=1.0 / 3.0,
                 min_stddev=0.0,
                 stddev_anneal_schedule=None,
                 stddev_transform=torch.nn.functional.softplus):
        super().__init__()

        self.split_network = split_network
        self.hidden_size = hidden_size

        if split_network: 
            base_observation_space = OrderedDict()
            arm_observation_space = OrderedDict()

            if "base_proprioceptive" in observation_space.spaces: 
                base_observation_space["base_proprioceptive"] = observation_space.spaces["base_proprioceptive"]
                arm_observation_space["base_proprioceptive"] = observation_space.spaces["base_proprioceptive"]

            if "arm_proprioceptive" in observation_space.spaces: 
                arm_observation_space["arm_proprioceptive"] = observation_space.spaces["arm_proprioceptive"]

            if "rgb" in observation_space.spaces:
                base_observation_space["rgb"] = observation_space.spaces["rgb"]

            if "depth" in observation_space.spaces:
                base_observation_space["depth"] = observation_space.spaces["depth"]

            if "seg" in observation_space.spaces:
                base_observation_space["seg"] = observation_space.spaces["seg"]

            if "wrist_rgb" in observation_space.spaces:
                arm_observation_space["wrist_rgb"] = observation_space.spaces["wrist_rgb"]

            if "wrist_depth" in observation_space.spaces:
                arm_observation_space["wrist_depth"] = observation_space.spaces["wrist_depth"]

            if "wrist_seg" in observation_space.spaces:
                arm_observation_space["wrist_seg"] = observation_space.spaces["wrist_seg"]

            base_observation_space = gym.spaces.Dict(base_observation_space)
            arm_observation_space = gym.spaces.Dict(arm_observation_space)

            print("BASE OBSERVATION SPACE")
            print(base_observation_space)

            print("ARM OBSERVATION SPACE")
            print(arm_observation_space)

            self.base_net = Net(
                observation_space=base_observation_space,
                hidden_size=hidden_size,
                cnn_layers_params=cnn_layers_params
            )

            self.arm_net = Net(
                observation_space=arm_observation_space,
                hidden_size=hidden_size,
                cnn_layers_params=cnn_layers_params
            )

            self.stddev_anneal_schedule = stddev_anneal_schedule
            if stddev_anneal_schedule is not None:
                assert action_space.__class__.__name__ == "Box", "can only anneal std. dev. for continuous action space"
                assert initial_stddev >= min_stddev, "initial std. dev. should be >= min std. dev."
                self.log_initial_stddev = np.log(initial_stddev + EPS)
                self.log_min_stddev = np.log(min_stddev + EPS)

            self.base_action_space = gym.spaces.Box(shape=(2,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

            self.arm_action_space = gym.spaces.Box(shape=(7,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

            if action_space.__class__.__name__ == "Box":
                self.base_action_distribution = DiagGaussianNet(self.base_net.output_size,
                                                           2,
                                                           self.base_action_space,
                                                           squash_mean=True,
                                                           initial_stddev=initial_stddev,
                                                           min_stddev=min_stddev,
                                                           stddev_transform=stddev_transform)
                self.arm_action_distribution = DiagGaussianNet(self.arm_net.output_size,
                                                           7,
                                                           self.arm_action_space,
                                                           squash_mean=True,
                                                           initial_stddev=initial_stddev,
                                                           min_stddev=min_stddev,
                                                           stddev_transform=stddev_transform)

            # For discrete camera action
            self.use_camera_masks = use_camera_masks

            if self.use_camera_masks:
                self.camera_mask_distribution = CategoricalNet(self.base_net.output_size, camera_masks_dim)

            # Critic Layer
            self.goal_hidden_size = 128

            self.critic_linear = nn.Linear(2*self.hidden_size + self.goal_hidden_size, 1)
            self.goal_linear = nn.Linear(3, self.goal_hidden_size)

            # Init critic layer 
            nn.init.orthogonal_(self.critic_linear.weight, gain=1)
            nn.init.constant_(self.critic_linear.bias, val=0)

            self.train()

        else: 
            self.net = Net(
                observation_space=observation_space,
                hidden_size=hidden_size,
                cnn_layers_params=cnn_layers_params,
            )
            self.stddev_anneal_schedule = stddev_anneal_schedule
            if stddev_anneal_schedule is not None:
                assert action_space.__class__.__name__ == "Box", "can only anneal std. dev. for continuous action space"
                assert initial_stddev >= min_stddev, "initial std. dev. should be >= min std. dev."
                self.log_initial_stddev = np.log(initial_stddev + EPS)
                self.log_min_stddev = np.log(min_stddev + EPS)

            if action_space.__class__.__name__ == "Discrete":
                num_outputs = action_space.n
                self.action_distribution = CategoricalNet(self.net.output_size, num_outputs)
            elif action_space.__class__.__name__ == "Box":
                num_outputs = action_space.shape[0]
                self.action_distribution = DiagGaussianNet(self.net.output_size,
                                                           num_outputs,
                                                           action_space,
                                                           squash_mean=True,
                                                           initial_stddev=initial_stddev,
                                                           min_stddev=min_stddev,
                                                           stddev_transform=stddev_transform)
            elif action_space.__class__.__name__ == "MultiDiscrete":
                num_outputs = action_space.nvec
                self.action_distribution = MultiCategoricalNet(self.net.output_size, num_outputs)

            # For discrete camera action
            self.use_camera_masks = use_camera_masks

            if self.use_camera_masks:
                self.camera_mask_distribution = CategoricalNet(self.net.output_size, camera_masks_dim)

    def forward(self, *x):
        raise NotImplementedError

    def get_current_stddev(self, update):
        anneal_progress = min(1.0, (float(update) / self.stddev_anneal_schedule))
        log_stddev = self.log_initial_stddev - (self.log_initial_stddev - self.log_min_stddev) * anneal_progress
        return np.exp(log_stddev)

    def act(self, observations, base_rnn_hidden_states, arm_rnn_hidden_states, masks, deterministic=False, update=None):
        if self.split_network: 
            base_observations, arm_observations = self.split_observations(observations)

            base_actor_features, base_rnn_hidden_states = self.base_net(base_observations, base_rnn_hidden_states, masks)
            arm_actor_features, arm_rnn_hidden_states = self.arm_net(arm_observations, arm_rnn_hidden_states, masks)

            if self.stddev_anneal_schedule is not None:
                stddev = self.get_current_stddev(update)
                base_distribution = self.base_action_distribution(base_actor_features, stddev=stddev)
                arm_distribution = self.arm_action_distribution(arm_actor_features, stddev=stddev)
            else:
                base_distribution = self.base_action_distribution(base_actor_features)
                arm_distribution = self.arm_action_distribution(arm_actor_features)

            if deterministic:
                base_action = base_distribution.mode()
                arm_action = arm_distribution.mode()

            else:
                base_action = base_distribution.sample()
                arm_action = arm_distribution.sample()

            base_action_log_probs = base_distribution.log_probs(base_action, 0, 2)
            arm_action_log_probs = arm_distribution.log_probs(arm_action, 0, 7)

            if self.use_camera_masks:
                camera_mask_distribution = self.camera_mask_distribution(base_actor_features)
                if deterministic: 
                    camera_mask_indices = camera_mask_distribution.mode()
                else: 
                    camera_mask_indices = camera_mask_distribution.sample()
                camera_mask_log_probs = camera_mask_distribution.log_probs(camera_mask_indices)
            else: 
                camera_mask_indices = torch.zeros_like(base_action_log_probs, dtype=torch.long)
                camera_mask_log_probs = torch.zeros_like(base_action_log_probs)

            value = self.get_value(observations, base_rnn_hidden_states, arm_rnn_hidden_states, masks)

            close_to_goal = observations['close_to_goal']

            if close_to_goal: 
                action = arm_action
            else: 
                action = torch.cat((base_action, arm_action[:,2:]), dim=1)

            return value, action, close_to_goal, base_action_log_probs, arm_action_log_probs, camera_mask_indices, camera_mask_log_probs, base_rnn_hidden_states, arm_rnn_hidden_states

        else:
            value, actor_features, rnn_hidden_states = self.net(observations, rnn_hidden_states, masks)

            if self.stddev_anneal_schedule is not None:
                stddev = self.get_current_stddev(update)
                distribution = self.action_distribution(actor_features, stddev=stddev)
            else:
                distribution = self.action_distribution(actor_features)

            if deterministic:
                action = distribution.mode()
            else:
                action = distribution.sample()
            action_log_probs = distribution.log_probs(action)

            if self.use_camera_masks:
                camera_mask_distribution = self.camera_mask_distribution(actor_features)
                if deterministic: 
                    camera_mask_indices = camera_mask_distribution.mode()
                else: 
                    camera_mask_indices = camera_mask_distribution.sample()
                camera_mask_log_probs = camera_mask_distribution.log_probs(camera_mask_indices)
            else: 
                camera_mask_indices = torch.zeros_like(action_log_probs, dtype=torch.long)
                camera_mask_log_probs = torch.zeros_like(action_log_probs)

            return value, action, action_log_probs, camera_mask_indices, camera_mask_log_probs, rnn_hidden_states

    def get_value(self, observations, base_rnn_hidden_states, arm_rnn_hidden_states, masks):
        if self.split_network:
            base_observations, arm_observations = self.split_observations(observations)

            base_actor_features, base_rnn_hidden_states = self.base_net(base_observations, base_rnn_hidden_states, masks)
            arm_actor_features, arm_rnn_hidden_states = self.arm_net(arm_observations, arm_rnn_hidden_states, masks)

            goal_embedding = self.goal_linear(observations['goal'])

            return self.critic_linear(torch.cat((torch.cat((base_actor_features, arm_actor_features), dim=1), goal_embedding), dim=1))
            
        else:
            value, _, _ = self.net(observations, rnn_hidden_states, masks)
            return value

    def evaluate_actions(self, observations, base_rnn_hidden_states, arm_rnn_hidden_states, masks, action, close_to_goal, camera_mask_indices, update=None):
        if self.split_network:
            base_observations, arm_observations = self.split_observations(observations)

            base_actor_features, base_rnn_hidden_states = self.base_net(base_observations, base_rnn_hidden_states, masks)
            arm_actor_features, arm_rnn_hidden_states = self.arm_net(arm_observations, arm_rnn_hidden_states, masks)

            if self.stddev_anneal_schedule is not None:
                stddev = self.get_current_stddev(update)
                base_distribution = self.base_action_distribution(base_actor_features, stddev=stddev)
                arm_distribution = self.arm_action_distribution(arm_actor_features, stddev=stddev)
            else:
                base_distribution = self.base_action_distribution(base_actor_features)
                arm_distribution = self.arm_action_distribution(arm_actor_features)

            base_action = action[:, :2]
            arm_action = action

            base_action_log_probs = base_distribution.log_probs(base_action, 0, 2)
            arm_action_log_probs_base = arm_distribution.log_probs(arm_action, 0, 2)
            arm_action_log_probs_arm = arm_distribution.log_probs(arm_action, 2, 7)

            base_distribution_entropy = base_distribution.entropy()
            arm_distribution_entropy = arm_distribution.entropy()

            if self.use_camera_masks:
                camera_mask_distribution = self.camera_mask_distribution(base_actor_features)
                camera_mask_log_probs = camera_mask_distribution.log_probs(camera_mask_indices)
                camera_mask_dist_entropy = camera_mask_distribution.entropy()
            else: 
                camera_mask_log_probs = torch.zeros_like(base_action_log_probs)
                camera_mask_dist_entropy = torch.zeros_like(base_distribution_entropy)

            complete_action_log_probs = (1-close_to_goal)*base_action_log_probs + close_to_goal*arm_action_log_probs_base + arm_action_log_probs_arm + camera_mask_log_probs
            dist_entropy = base_distribution_entropy + arm_distribution_entropy + camera_mask_dist_entropy

            value = self.get_value(observations, base_rnn_hidden_states, arm_rnn_hidden_states, masks)

            return value, complete_action_log_probs, dist_entropy, base_rnn_hidden_states, arm_rnn_hidden_states

        else:
            value, actor_features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, masks
            )
            if self.stddev_anneal_schedule is not None:
                stddev = self.get_current_stddev(update)
                distribution = self.action_distribution(actor_features, stddev=stddev)
            else:
                distribution = self.action_distribution(actor_features)
            action_log_probs = distribution.log_probs(action)
            distribution_entropy = distribution.entropy()

            if self.use_camera_masks:
                camera_mask_distribution = self.camera_mask_distribution(actor_features)
                camera_mask_log_probs = camera_mask_distribution.log_probs(camera_mask_indices)
                camera_mask_dist_entropy = camera_mask_distribution.entropy()
            else: 
                camera_mask_log_probs = torch.zeros_like(action_log_probs)
                camera_mask_dist_entropy = torch.zeros_like(distribution_entropy)

            complete_action_log_probs = action_log_probs + camera_mask_log_probs
            dist_entropy = distribution_entropy + camera_mask_dist_entropy

            return value, complete_action_log_probs, dist_entropy, rnn_hidden_states

    def split_observations(self, observations):
        base_observations = {}
        arm_observations = {}

        if "base_proprioceptive" in observations: 
            base_observations['base_proprioceptive'] = observations['base_proprioceptive']
            arm_observations['base_proprioceptive'] = observations['base_proprioceptive']

        if "arm_proprioceptive" in observations: 
            arm_observations['arm_proprioceptive'] = observations['arm_proprioceptive']

        if "rgb" in observations:
            base_observations["rgb"] = observations["rgb"]

        if "depth" in observations:
            base_observations["depth"] = observations["depth"]

        if "seg" in observations:
            base_observations["seg"] = observations["seg"]

        if "wrist_rgb" in observations:
            arm_observations["wrist_rgb"] = observations["wrist_rgb"]

        if "wrist_depth" in observations:
            arm_observations["wrist_depth"] = observations["wrist_depth"]

        if "wrist_seg" in observations:
            arm_observations["wrist_seg"] = observations["wrist_seg"]

        return base_observations, arm_observations
