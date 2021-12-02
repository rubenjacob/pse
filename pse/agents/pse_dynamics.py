import random
from pathlib import Path
from typing import Any, Dict, Iterator, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from pse.agents import utils
from pse.agents.drq import DrQV2Agent
from pse.agents.pse_drq import PSEDrQAgent
from pse.agents.utils import load_snapshot_payload


class ProbabilisticTransitionModel(nn.Module):
    def __init__(self, encoder_feature_dim, action_shape, layer_width, announce=True, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.fc = nn. Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class EnsembleOfProbabilisticTransitionModels(object):
    def __init__(self, encoder_feature_dim, action_shape, layer_width=512, ensemble_size=5):
        self.models = [ProbabilisticTransitionModel(encoder_feature_dim, action_shape, layer_width, announce=False)
                       for _ in range(ensemble_size)]
        print("Ensemble of probabilistic transition models chosen.")

    def __call__(self, x):
        mu_sigma_list = [model.forward(x) for model in self.models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        return mus, sigmas

    def sample_prediction(self, x):
        model = random.choice(self.models)
        return model.sample_prediction(x)

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def parameters(self):
        list_of_parameters = [list(model.parameters()) for model in self.models]
        parameters = [p for ps in list_of_parameters for p in ps]
        return parameters


class PSEDynamicsAgent(PSEDrQAgent):
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_wandb, contrastive_loss_weight,
                 contrastive_loss_temperature, metric_data_dir, optimal_policy_dir):
        super().__init__(obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau,
                         num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_wandb,
                         contrastive_loss_weight, contrastive_loss_temperature, metric_data_dir)

        self._transition_model = EnsembleOfProbabilisticTransitionModels(feature_dim, action_shape).to(device)
        self._transition_optimizer = torch.optim.Adam(self._transition_model.parameters(), lr=lr)

        self._optimal_policy = self.initialize_optimal_policy(optimal_policy_dir)

    def initialize_optimal_policy(self, optimal_policy_dir: str) -> Callable[[torch.Tensor], torch.Tensor]:
        agent: DrQV2Agent = load_snapshot_payload(snapshot_dir=Path(optimal_policy_dir), device=self.device)['agent']

        def optimal_policy(h: torch.Tensor) -> torch.Tensor:
            mu = agent.actor.policy(h)
            action = torch.tanh(mu)

            with torch.no_grad():
                return action

        return optimal_policy

    def update_transition_model(self, action: torch.Tensor, obs: torch.Tensor,
                                next_obs: torch.Tensor) -> Dict[str, Any]:
        metrics = dict()

        h = self._encode_obs(obs)
        next_h = self._encode_obs(next_obs)

        pred_next_latent_mu, pred_next_latent_sigma = self._transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

        self._transition_optimizer.zero_grad(set_to_none=True)
        transition_loss.backward()
        self._transition_optimizer.step()

        metrics['transition_loss'] = transition_loss
        return metrics

    def calculate_metric_vals(self, obs1: torch.Tensor, obs2: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        h1 = self._encode_obs(obs1)
        h2 = self._encode_obs(obs2)
        action1 = self._optimal_policy(h1)
        action2 = self._optimal_policy(h2)
        pred_next_latent_mu1, pred_next_latent_sigma1 = self._transition_model(torch.cat([h1, action1], dim=1))
        pred_next_latent_mu2, pred_next_latent_sigma2 = self._transition_model(torch.cat([h2, action2], dim=1))
        pred_next_latent_mu1, pred_next_latent_sigma1 = pred_next_latent_mu1.mean(dim=0), pred_next_latent_sigma1.mean(dim=0)
        pred_next_latent_mu2, pred_next_latent_sigma2 = pred_next_latent_mu2.mean(dim=0), pred_next_latent_sigma2.mean(dim=0)

        action_diff = action1.unsqueeze(1) - action2.unsqueeze(0)
        action_dist = torch.mean(torch.abs(action_diff), dim=-1)
        transition_dist = torch.mean(torch.sqrt(
            (pred_next_latent_mu1.unsqueeze(1) - pred_next_latent_mu2.unsqueeze(0)).pow(2) +
            (pred_next_latent_sigma1.unsqueeze(1) - pred_next_latent_sigma2.unsqueeze(0)).pow(2)
        ), dim=-1)
        return action_dist + 0.99 * transition_dist

    def update(self, replay_iter: Iterator[DataLoader], step: int) -> Dict[str, Any]:
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_transition_model(action, obs, next_obs))

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        episode = next(self.metric_data_iter)[:2]
        obs1, obs2 = utils.to_torch(episode, self.device)
        obs1, obs2 = torch.squeeze(obs1.float()), torch.squeeze(obs2.float())
        obs1, obs2 = self.aug(obs1), self.aug(obs2)

        metric_vals = self.calculate_metric_vals(obs1, obs2, discount)

        self._contrastive_optimizer.zero_grad()
        contr_loss = self._contrastive_loss_weight * self.representation_alignment_loss(obs1, obs2, metric_vals)

        metrics['contrastive_loss'] = contr_loss

        contr_loss.backward()
        self._contrastive_optimizer.step()

        return metrics
