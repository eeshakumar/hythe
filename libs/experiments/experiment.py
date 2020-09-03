from os import path, listdir
from json import load as json_load
import glob

from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_project.bark.runtime.scenario.scenario_generation.config_with_ease import \
    ConfigWithEase

from hythe.libs.blueprint.blueprint import HyHighwayDiscreteBlueprint
from hythe.libs.environments.gym import HyDiscreteHighway


class Experiment(object):

    def __init__(self, params, agent, scenario_generation=None):
        self._agent = agent
        self._env = agent.env
        self._blueprint = self._env.blueprint
        self._scenario_generation = self._blueprint.scenario_generation
        self._params = params
        self.init_params(params)

    def init_params(self, params):
        if params is not None:
            self.num_episodes = params["Experiment"]["num_episodes"]
            self.dir = params["Experiment"]["dir"]
            self.params_file = params["Experiment"]["params"]
            self.scnearios_generated_file = \
                params["Experiment"]["scenarios_generated"]
            self.random_seed = params["Experiment"]["random_seed"]
        else:
            raise IOError("Params Configuration not found.")

    def update_filenames(self, episode_num):
        params_filename = self.params_file.format(self.random_seed, episode_num)
        scenarios_filename = self.scnearios_generated_file.format(self.random_seed, episode_num)
        self._blueprint.scenario_generation.params["Experiment"]["params"] = \
            params_filename
        self._blueprint.scenario_generation.params["Experiment"]["scenarios_generated"] = \
            scenarios_filename
        scenarios_filename = path.join(self.dir, scenarios_filename)
        params_filename = path.join(self.dir, params_filename)
        return scenarios_filename, params_filename

    def get_behavior(self):
        return

    def save(self, episode_num):
        try:
            if episode_num % 1000 == 0:
                scenarios_filename, params_filename = self.update_filenames(episode_num=episode_num)
                self._blueprint.scenario_generation.dump_scenario_list(scenarios_filename)

                self._blueprint.scenario_generation.params.Save(params_filename)
                self._agent.save_models(self.dir)
        except TypeError or IOError as error:
            print("Could not save experiment:", error)

    @staticmethod
    def load(exp_dir):
        params_files = glob.glob(path.join(exp_dir, "params*.json"))
        scenario_files = glob.glob(path.join(exp_dir, "scenarios_list*"))
        assert len(params_files) == len(scenario_files)
        return params_files, scenario_files

    def prune(self):
        # Dont save all scenarios, save last x scenarios
        return

    @staticmethod
    def is_experiment(exp_dir):
        assert path.exists(exp_dir)
        assert len(listdir(exp_dir)) > 0
        return True

    @property
    def agent(self):
        return self._agent

    @property
    def env(self):
        return self._env

    @property
    def blueprint(self):
        return self._blueprint

    def run_single_episode(self, episode_num=1, is_loaded=False):
        self._agent.train_episode()
        if not is_loaded:
            self.save(episode_num)
        return

    def run(self, only_one=False):
        if only_one:
            self.run_single_episode()
        else:
            for i in range(1, self.num_episodes + 1):
                self.run_single_episode(i)

    @staticmethod
    def restore_env(params, blueprint):
        return HyDiscreteHighway(params=params, blueprint=blueprint)

    @staticmethod
    def load_json(filename):
        with open(filename, "r+") as json_file:
            data = json_load(json_file)
            return data

    @staticmethod
    def restore_blueprint(some_params,
                          scenario_generated_file,
                          num_scenarios=25,
                          behavior=None,
                          evaluator=None,
                          observer=None):

        if isinstance(some_params, ParameterServer):
            params = some_params
        else:
            json_data = Experiment.load_json(some_params)
            json_params = {"json": json_data}
            params = ParameterServer(**json_params)
        map_file_name = params["Experiment"]["map_filename"] or \
                        "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
        scenario_generation = ConfigWithEase(num_scenarios, params=params,
                                             map_file_name=map_file_name)
        scenario_generation.load_scenario_list(scenario_generated_file)
        return params, HyHighwayDiscreteBlueprint(params=params,
                                                  scenario_generation=scenario_generation,
                                                  behavior=behavior,
                                                  evaluator=evaluator,
                                                  observer=observer)
