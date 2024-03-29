import hydra

from pse.agents.train import Workspace
from pse.agents.utils import set_seed_everywhere


@hydra.main(config_path='../configs', config_name='optimal_policy_config.yaml')
def main(cfg):
    set_seed_everywhere(seed=cfg.seed)

    workspace = Workspace(cfg)
    if cfg.resume_from_snapshot:
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
