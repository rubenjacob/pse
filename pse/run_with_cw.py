from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from pse.run import main


class ClusterWorkExperiment(experiment.AbstractExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        main()

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    # expects the class not an instance!
    cw = cluster_work.ClusterWork(ClusterWorkExperiment)
    cw.run()
