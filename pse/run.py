import os

import hydra
import numpy as np
import torch

from pse.agents.train_drq import Workspace


@hydra.main(config_path='configs/config.yaml', strict=True)
def main(cfg):
    # set random seed everywhere
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    workspace = Workspace(cfg)
    if cfg.resume_from_snapshot:
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
