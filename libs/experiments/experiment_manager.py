import uuid
import yaml
from pathlib import Path
from argparse import ArgumentParser
from fqf_iqn_qrdqn.agent import FQFAgent
import os
from datetime import datetime

from hythe.libs.environments.gym import HyDiscreteHighway
from hythe.libs.experiments.experiment import Experiment


class ExperimentManager(object):

    def __init__(self, num_experiments=2, num_scenarios=10, random_seed=0,
                 params_list=None, params_files=None):
        # Move assertion
        # if params_list, then launch new experiments, else restore blueprints.
        assert any([params_files, params_list]) is not None
        self.num_experiments = num_experiments
        if params_list is not None:
            self._params_list = self.configure_params(params_list)
        elif params_files is not None:
            self._params_list = self.configure_params(is_files=True)

        self._num_scenarios = num_scenarios
        self.random_seed = random_seed
        # self._scenarios_dict = self.init_scenarios()
        self._experiments = self.init_experiments()
        self.init_experiments()

    def init_scenarios(self):
        # For LOADED EXPERIMENTS
        scenarios_generated_dict = []
        for params in self._params_list:
            scenarios_generated_dict[params["Experiment"]["random_seed"]] = \
                params["Experiment"]["scenarios_generated"]
        return scenarios_generated_dict

    def configure_args(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        parser.add_argument(
            '--config', type=str, default=os.path.join('../fqn/config', 'fqf.yaml'))
        parser.add_argument('--env_id', type=str, default='hyhighway-v0')
        parser.add_argument('--cuda', action='store_true', default=True)
        parser.add_argument('--seed', type=int, default=122)
        return parser.parse_args()

    def configure_agent(self, env):
        args = self.configure_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        name = args.config.split('/')[-1].rstrip('.yaml')
        time = datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = os.path.join(
            'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
        agent = FQFAgent(env=env, test_env=env, log_dir=log_dir, seed=args.seed,
                         cuda=args.cuda, **config)
        return agent

    def configure_params(self, some_params, is_files=False):
        # TODO: Move this to commons
        if not is_files:
            for params in some_params:
                experiment_seed = str(uuid.uuid4())
                params["Experiment"]["random_seed"] = experiment_seed
                params["Experiment"]["dir"] = "/home/ekumar/master_thesis/code/hythe/output/experiments/exp_{}".format(
                    experiment_seed)
                Path(params["Experiment"]["dir"]).mkdir(parents=True, exist_ok=True)
                params["Experiment"]["params"] = "params_{}_{}.json"
                params["Experiment"]["scenarios_generated"] = "scenarios_list_{}_{}"
                params["Experiment"]["num_episodes"] = 10
                params["Experiment"][
                    "map_filename"] = "bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
                print("Configured params with random seed:", experiment_seed)
            return some_params
        else:
            params_list = []
            for params_file in some_params:
                break
                # Do some init for each file json.load()
            return params_list

    def init_experiments(self):
        experiments = {}
        for params in self._params_list:
            env = HyDiscreteHighway(params=params, num_scenarios=self._num_scenarios,
                                    random_seed=self.random_seed,
                                    map_filename=params["Experiment"]["map_filename"])
            agent = self.configure_agent(env=env)
            experiments[params["Experiment"]["random_seed"]] = Experiment(params, agent)
        return experiments

    def dispatch(self):
        from multiprocessing import Process
        for (seed, experiment) in self._experiments.items():
            p = Process(target=self.run, args=(experiment, ))
            p.start()
            p.join()
        return

    def run(self, experiment):
        # method/process for single experiment execution
        print("Running Experiment with seed:", experiment.random_seed)
        experiment.run()
        return

    @property
    def experiments(self):
        return self._experiments
