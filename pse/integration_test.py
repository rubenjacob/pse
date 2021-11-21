import hydra

from pse.agents.train_drq import Workspace


@hydra.main(config_path="../configs/test_config.yaml")
def check_workspace_builds_and_forward_pass_works(cfg):
    print(cfg)
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    check_workspace_builds_and_forward_pass_works()
