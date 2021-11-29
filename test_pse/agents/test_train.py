import uuid

import wandb


def test_wandb_resuming():
    run_name = "test_resuming"
    run_id = str(uuid.uuid4())
    wandb.init(project="tests", name=run_name, id=run_id)
    for step in range(5):
        wandb.log({"loss": 1.0 / (step + 1)})
    wandb.finish()

    wandb.init(project="tests", name=run_name, resume="must", id=run_id)
    for step in range(5, 10):
        wandb.log({"loss": 1.0 / (step + 1)})

    assert wandb.run.resumed
