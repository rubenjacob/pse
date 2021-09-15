from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import hydra


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.shape[0], -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.encoder: Encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim, 2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    """Critic network, employs double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder: Encoder = hydra.utils.instantiate(encoder_cfg)

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, action_shape, action_range, device, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau, critic_target_update_frequency, batch_size):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.training = True

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.aug = utils.RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, eval_mode=True):
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()

        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug, action, reward, next_obs, next_obs_aug):
        metrics: Dict[str, Any] = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + self.discount * target_V

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug, next_action_aug)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + self.discount * target_V

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = current_Q1.mean().item()
        metrics['critic_q2'] = current_Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return metrics

    def update_actor_and_alpha(self, obs):
        metrics: Dict[str, Any] = dict()

        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['target_ent'] = self.target_entropy

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha_value'] = self.alpha

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return metrics

    def update(self, replay_iter, step):
        metrics: Dict[str, Any] = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        obs_aug = self.aug(obs.float())
        next_obs_aug = self.aug(next_obs.float())
        metrics['train/batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(obs, obs_aug, action, reward, next_obs, next_obs_aug))

        if step % self.actor_update_frequency == 0:
            metrics.update(self.update_actor_and_alpha(obs))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return metrics
