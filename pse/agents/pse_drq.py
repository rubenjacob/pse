from typing import Union, Tuple, Dict, Any, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from pse.agents import utils
from pse.agents.drq import DrQV2Agent
from pse.utils.helper_functions import torch_gather_nd, torch_scatter_nd_update, EPS, sample_indices, cosine_similarity


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


class PSEDrQAgent(DrQV2Agent):
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_wandb, contrastive_loss_weight=1.0,
                 contrastive_loss_temperature=0.1):
        super().__init__(obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau,
                         num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_wandb)
        self._contrastive_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self._contrastive_loss_weight = contrastive_loss_weight
        self._contrastive_loss_temperature = contrastive_loss_temperature

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device)
        encoded = self.encoder(obs.unsqueeze(0))
        return self.actor.trunk(encoded)

    def contrastive_metric_loss(self, obs1: torch.Tensor, obs2: torch.Tensor, metric_vals: torch.Tensor,
                                use_coupling_weights: bool = False,
                                coupling_temperature: float = 0.1, return_representation: bool = False,
                                temperature: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if np.random.randint(2) == 1:
            obs2, obs1 = obs1, obs2
            metric_vals.transpose(0, 1)

        indices = sample_indices(metric_vals.size()[0], sort=return_representation)
        obs1 = torch.gather(obs1, index=indices, dim=0)
        metric_vals = torch.gather(metric_vals, index=indices, dim=0)

        representation_1 = self._encode_obs(obs1)
        representation_2 = self._encode_obs(obs2)
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

    def update(self, replay_iter: Iterator[DataLoader], step: int,
               metric_data_iter: Optional[Iterator[DataLoader]] = None):
        metrics: Dict[str, Any] = super(PSEDrQAgent, self).update(replay_iter, step)

        assert metric_data_iter is not None
        episode = next(metric_data_iter)
        obs1, obs2, metric_vals = utils.to_torch(episode, self.device)
        obs1, obs2 = self.aug(obs1), self.aug(obs2)

        self._contrastive_optimizer.zero_grad()
        contr_loss = self._contrastive_loss_weight * self.contrastive_metric_loss(obs1, obs2, metric_vals)

        metrics['contrastive_loss'] = contr_loss

        contr_loss.backward()
        self._contrastive_optimizer.step()

        return metrics
