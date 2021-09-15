from typing import Union, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from pse.agents import utils
from pse.agents.drq import DRQAgent
from pse.utils.helper_functions import torch_gather_nd, torch_scatter_nd_update, EPS, sample_indices, cosine_similarity


def metric_fixed_point_fast(cost_matrix, gamma=0.99, eps=1e-7):
    """Dynamic programming for calculating PSM."""
    d = np.zeros_like(cost_matrix)
    def operator(d_cur):
        d_new = 1 * cost_matrix
        discounted_d_cur = gamma * d_cur
        d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
        d_new[:-1, -1] += discounted_d_cur[1:, -1]
        d_new[-1, :-1] += discounted_d_cur[-1, 1:]
        return d_new

    while True:
        d_new = operator(d)
        if np.sum(np.abs(d - d_new)) < eps:
            break
        else:
            d = d_new[:]
    return d


def contrastive_loss(similarity_matrix: torch.Tensor, metric_vals: torch.Tensor, temperature: float,
                     coupling_temperature: float, use_coupling_weights: bool) -> torch.Tensor:
    similarity_matrix /= temperature
    neg_logits1 = similarity_matrix

    col_indices = torch.argmin(metric_vals, dim=1)
    pos_indices1 = torch.stack([torch.range(0, metric_vals.size()[0]), col_indices], dim=1)
    pos_logits1 = torch_gather_nd(similarity_matrix, pos_indices1)

    if use_coupling_weights:
        metric_vals /= coupling_temperature
        coupling = torch.exp(-metric_vals)
        pos_weights1 = -torch_gather_nd(metric_vals, pos_indices1)
        pos_logits1 += pos_weights1
        neg_weights = torch.log((1.0 - coupling) + EPS)
        neg_logits1 += torch_scatter_nd_update(neg_weights, pos_indices1, pos_weights1)

    neg_logits1 = torch.logsumexp(neg_logits1, dim=1)
    return torch.mean(neg_logits1 - pos_logits1)


class PSEDRQAgent(DRQAgent):
    def __init__(self, action_shape, action_range, device, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau, critic_target_update_frequency, batch_size,
                 contrastive_loss_weight, contrastive_loss_temperature):
        super().__init__(action_shape, action_range, device, critic_cfg, actor_cfg, discount,
                         init_temperature, lr, actor_update_frequency, critic_tau, critic_target_update_frequency,
                         batch_size)
        self._contrastive_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self._contrastive_loss_weight = contrastive_loss_weight
        self._contrastive_loss_temperature = contrastive_loss_temperature

    def contrastive_metric_loss(self, obs1, obs2, metric_vals: torch.Tensor, use_coupling_weights: bool = False,
                                coupling_temperature: float = 0.1, return_representation: bool = False,
                                temperature: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if np.random.randint(2) == 1:
            obs2, obs1 = obs1, obs2
            metric_vals.transpose(0, 1)

        indices = sample_indices(metric_vals.size()[0], sort=return_representation)
        obs1 = torch.gather(obs1, index=indices, dim=0)
        metric_vals = torch.gather(metric_vals, index=indices, dim=0)

        representation_1 = self.actor(obs1)
        representation_2 = self.actor(obs2)
        similarity_matrix = cosine_similarity(representation_1, representation_2)
        alignment_loss = contrastive_loss(similarity_matrix=similarity_matrix,
                                          metric_vals=metric_vals,
                                          temperature=temperature,
                                          use_coupling_weights=use_coupling_weights,
                                          coupling_temperature=coupling_temperature)

        if return_representation:
            return alignment_loss, similarity_matrix
        else:
            return alignment_loss

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

        self._contrastive_optimizer.zero_grad()
        contr_loss = self._contrastive_loss_weight * self.contrastive_metric_loss(obs1, obs2, metric_vals)

        metrics['contrastive_loss'] = contr_loss

        contr_loss.backward()
        self._contrastive_optimizer.step()
