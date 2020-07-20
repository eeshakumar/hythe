

class Experiment(object):

    def __init__(self, scenario_generation, agent):
        self._scenario_generation = scenario_generation
        self._agent = agent
        self._exp_dir = exp_dir

    def get_behavior(self):
        return

    def run_single_experiment(self):
        return

    def save(self):
        return

    @staticmethod
    def is_experiment(exp_dir):
        return

    @property
    def scenario_generation(self):
        return self._scenario_generation

    @property
    def agent(self):
        return self._agent

