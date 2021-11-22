import sys

import hydra

from pse.agents.train_drq import Workspace


@hydra.main(config_path="../configs", config_name="test_config.yaml")
def check_workspace_builds_and_forward_pass_works(cfg):
    print(cfg)
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    print(sys.argv[1:])
    check_workspace_builds_and_forward_pass_works()
