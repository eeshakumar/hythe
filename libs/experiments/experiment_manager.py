import uuid
from collections import OrderedDict

import yaml
from pathlib import Path
from argparse import ArgumentParser
from fqf_iqn_qrdqn.agent import FQFAgent
import os
from datetime import datetime

from bark_project.modules.runtime.commons.parameters import ParameterServer
from hythe.libs.dispatch.sequential_dispatcher import SequentialDispatcher
from hythe.libs.environments.gym import HyDiscreteHighway
from hythe.libs.experiments.experiment import Experiment


class ExperimentManager(object):

    def __init__(self, dispatcher=None, num_experiments=2, num_scenarios=25, random_seed=0,
                 params_list=None, params_files=None):
        if dispatcher is None:
            self._dispatcher = SequentialDispatcher()
        else:
            self._dispatcher = dispatcher
        # Move assertion
        # if params_list, then launch new experiments, else restore blueprints.
        assert any([params_files, params_list]) is not None, "Either params_files or params_list is needed as input"
        self.num_experiments = num_experiments
        if params_list is not None:
            self._params_list = self.configure_params(params_list)
            self._scenarios_generated_dict = None
        elif params_files is not None:
            self._params_list = self.configure_params(params_files, is_files=True)
            self._scenarios_generated_dict = self.init_scenarios()

        self._num_scenarios = num_scenarios
        self.random_seed = random_seed
        self._experiments = self.init_experiments()
        self._dispatcher.set_dispatch_dict(self._experiments)
        self.init_experiments()
        if num_experiments != len(self._experiments):
            self.num_experiments = len(self._experiments)
        self.experiment_process_list = []

    def init_scenarios(self):
        # For LOADED EXPERIMENTS
        scenarios_generated_dict = {}
        for params in self._params_list:
            scenarios_generated_dict[params["Experiment"]["random_seed"]] = \
                os.path.join(params["Experiment"]["dir"], params["Experiment"]["scenarios_generated"])
        return scenarios_generated_dict


    # def init_scenario_generation(self):
    #     if self._scenarios_generated_dict is not None:
    #         for (key, value) in self._scenarios_generated_dict:
    #             scenario_path = os.path.join(key, value)
    #             scenario_generation =


    def configure_args(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        parser.add_argument(
            '--config', type=str, default=os.path.join('hy-x-run.runfiles/fqn/config', 'fqf.yaml'))
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
                params["Experiment"]["dir"] = str(Path.home().joinpath(".cache/output/experiments/exp_{}".format(
                    experiment_seed)))
                Path(params["Experiment"]["dir"]).mkdir(parents=True, exist_ok=True)
                params["Experiment"]["params"] = "params_{}_{}.json"
                params["Experiment"]["scenarios_generated"] = "scenarios_list_{}_{}"
                params["Experiment"]["num_episodes"] = 10000
                params["Experiment"][
                    "map_filename"] = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
                print("Configured params with random seed:", experiment_seed)
            return some_params
        else:
            params_list = []
            exp_seeds = []
            for params_file in some_params:
                json_data = Experiment.load_json(params_file)
                exp_seed = json_data["Experiment"]["random_seed"]
                if exp_seed not in exp_seeds:
                    print("Adding seed:", exp_seed)
                    exp_seeds.append(exp_seed)
                    json_params = {"json": json_data}
                    params = ParameterServer(**json_params)
                    params["Experiment"]["params"] = "params_{}_{}.json"
                    params_list.append(params)
            return params_list

    def init_experiments(self):
        experiments = OrderedDict()
        for params in self._params_list:
            exp_seed = params["Experiment"]["random_seed"]
            if self._scenarios_generated_dict is not None:
                params, blueprint = Experiment.restore_blueprint(params, self._scenarios_generated_dict[exp_seed])
                params["Experiment"]["scenarios_generated"] = "scenarios_list_{}_{}"
                env = Experiment.restore_env(params=params, blueprint=blueprint)
            else:
                env = HyDiscreteHighway(params=params, num_scenarios=self._num_scenarios,
                                    random_seed=self.random_seed,
                                    map_filename=params["Experiment"]["map_filename"])
            agent = self.configure_agent(env=env)
            experiments[exp_seed] = Experiment(params, agent)
        return experiments

    def run_experiments(self):
        self._dispatcher.dispatch()

    @property
    def dispatcher(self):
        return self._dispatcher

    @property
    def experiments(self):
        return self._experiments
