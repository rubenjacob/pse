from typing import Union, Tuple, Dict, Any, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from pse.agents import utils
from pse.agents.drq import DrQV2Agent
from pse.data.metric_dataset import make_metric_data_loader
from pse.utils.helper_functions import sample_indices, cosine_similarity, contrastive_loss


class PSEDrQAgent(DrQV2Agent):
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_wandb, contrastive_loss_weight,
                 contrastive_loss_temperature, metric_data_dir):
        super().__init__(obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau,
                         num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_wandb)
        self._contrastive_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self._contrastive_loss_weight = contrastive_loss_weight
        self._contrastive_loss_temperature = contrastive_loss_temperature
        self.metric_data_loader = make_metric_data_loader(data_dir=metric_data_dir, num_workers=4)
        self._metric_data_iter = None

    @property
    def metric_data_iter(self) -> Optional[Iterator[DataLoader]]:
        if self.metric_data_loader is None:
            return None
        if self._metric_data_iter is None:
            self._metric_data_iter = iter(self.metric_data_loader)
        return self._metric_data_iter

    def __getstate__(self):
        """Make the agent picklable"""
        d = dict(self.__dict__)
        d['_metric_data_iter'] = None
        return d

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device)
        encoded = self.encoder(obs)
        return self.actor.trunk(encoded)

    def representation_alignment_loss(self, obs1: torch.Tensor, obs2: torch.Tensor, metric_vals: torch.Tensor,
                                      use_coupling_weights: bool = True, coupling_temperature: float = 0.1,
                                      return_representation: bool = False, temperature: float = 1.0) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if np.random.randint(2) == 1:
            obs2, obs1 = obs1, obs2
            metric_vals.transpose(0, 1)

        indices = sample_indices(metric_vals.size()[0], sort=return_representation)
        obs1 = obs1[indices]
        metric_vals = metric_vals[indices]

        representation_1 = self._encode_obs(obs1)
        representation_2 = self._encode_obs(obs2)
        similarity_matrix = cosine_similarity(representation_1, representation_2)
        alignment_loss = contrastive_loss(similarity_matrix=similarity_matrix,
                                          metric_vals=metric_vals,
                                          temperature=temperature,
                                          use_coupling_weights=use_coupling_weights,
                                          coupling_temperature=coupling_temperature,
                                          device=self.device)

        if return_representation:
            return alignment_loss, similarity_matrix
        else:
            return alignment_loss

    def update(self, replay_iter: Iterator[DataLoader], step: int) -> Dict[str, Any]:
        metrics: Dict[str, Any] = super(PSEDrQAgent, self).update(replay_iter, step)

        episode = next(self.metric_data_iter)
        obs1, obs2, metric_vals = utils.to_torch(episode, self.device)
        obs1, obs2, metric_vals = torch.squeeze(obs1.float()), torch.squeeze(obs2.float()), \
            torch.squeeze(metric_vals.float())
        obs1, obs2 = self.aug(obs1), self.aug(obs2)

        self._contrastive_optimizer.zero_grad()
        contr_loss = self._contrastive_loss_weight * self.representation_alignment_loss(obs1, obs2, metric_vals)

        metrics['contrastive_loss'] = contr_loss

        contr_loss.backward()
        self._contrastive_optimizer.step()

        return metrics
