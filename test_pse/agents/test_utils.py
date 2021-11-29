from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, output_size):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(OrderedDict([("linear1", nn.Linear(10, output_size)),
                                              ("act1", nn.ReLU())]))

    def forward(self, inputs):
        return self.net(inputs)


class Wrapper:
    def __init__(self, output_size):
        self.output_size = output_size
        self.model = SimpleModel(output_size=self.output_size)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.model.train()

    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.unsqueeze(0)
        output = self.model(obs)

        with torch.no_grad():
            return output.cpu().numpy()[0]

    def update(self, obs, target):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        pred = self.model(obs)
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()


def test_model_saving():
    wrapper = Wrapper(output_size=5)
    obs = np.ones((2, 10))
    target = np.zeros((2, 5))

    # do some train steps
    for step in range(10):
        wrapper.update(obs=obs, target=target)

    # save
    torch.save({'agent': wrapper}, str(Path.cwd() / 'snapshot.pt'))


def test_model_loading():
    obs = np.zeros((2, 10))
    loaded_agent: Wrapper = torch.load(str(Path.cwd() / 'snapshot.pt'))['agent']
    action = loaded_agent.act(obs)
    assert action.shape == (2, 5)


def test_model_state_saving():
    loaded_agent: Wrapper = torch.load(str(Path.cwd() / 'snapshot.pt'))['agent']
    torch.save(loaded_agent.model.state_dict())
