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
                 camera_masks_dim=3,
                 hidden_size=512,
                 cnn_layers_params=None,
                 initial_stddev=1.0 / 3.0,
                 min_stddev=0.0,
                 stddev_anneal_schedule=None,
                 stddev_transform=torch.nn.functional.softplus):
        super().__init__()
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

    def act(self, observations, rnn_hidden_states, masks, deterministic=False, update=None):
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

    def get_value(self, observations, rnn_hidden_states, masks):
        value, _, _ = self.net(observations, rnn_hidden_states, masks)
        return value

    def evaluate_actions(self, observations, rnn_hidden_states, masks, action, camera_mask_indices, update=None):
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
