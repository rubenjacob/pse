import hydra

from pse.agents.train_drq import Workspace
from pse.agents.utils import set_seed_everywhere


@hydra.main(config_path='../configs/config.yaml', strict=True)
def main(cfg):
    set_seed_everywhere(seed=cfg.seed)

    workspace = Workspace(cfg)
    if cfg.resume_from_snapshot:
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()