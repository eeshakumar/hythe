

class Experiment(object):

    def __init__(self, params=None, scenario_generation=None, agent=None, exp_dir=None):
        self._scenario_generation = scenario_generation
        self._agent = agent
        self._exp_dir = exp_dir
        self._env = agent.env

        self.init_params(params)

    def init_params(self, parms=None):
        self.no_episodes = 100

    def get_behavior(self):
        return

    def run_single_experiment(self):
        return

    def save(self):
        return

    @staticmethod
    def is_experiment(exp_dir):
        # run random checks in dir to verify it is experiment dir
        return

    @property
    def scenario_generation(self):
        return self._scenario_generation

    @property
    def agent(self):
        return self._agent

    @property
    def env(self):
        return self._env

    def run_single_episode(self):
        self._agent.run()
        return

    def run(self):
        for i in range(0, self.no_episodes):
            self.run_single_episode()

    def save(self):
        return
