

class Experiment(object):

    def __init__(self, params, agent):
        self._agent = agent
        self._env = agent.env
        self._blueprint = self._env.blueprint
        self.init_params(params)

    def init_params(self, params):
        self.no_episodes = 10

    def get_behavior(self):
        return

    def run_single_experiment(self):
        return

    def save(self):
        return

    def load(self):
        return

    @staticmethod
    def is_experiment(exp_dir):
        # run random checks in dir to verify it is experiment dir
        return

    @property
    def agent(self):
        return self._agent

    @property
    def env(self):
        return self._env

    @property
    def blueprint(self):
        return self._blueprint

    def run_single_episode(self):
        self._agent.train_episode()
        return

    def run(self):
        for i in range(0, self.no_episodes):
            self.run_single_episode()
