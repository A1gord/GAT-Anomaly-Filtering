from .execution import ExperimentRunner


def run_experiments(config):
    return ExperimentRunner(config).run()
