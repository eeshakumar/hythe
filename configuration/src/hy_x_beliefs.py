
try:
    import debug_settings
except:
    pass
import glob
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info("Running on process with ID: {}".format(os.getpid()))
import bark.core.commons
import bark.core
import bark.core.models.behavior
from bark.runtime.commons.parameters import ParameterServer

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from hythe.libs.observer.belief_observer import BeliefObserver
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
import yaml
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import ConfigurableScenarioGeneration

from hythe.libs.experiments.experiment import Experiment
from hythe.libs.environments.gym import HyDiscreteHighway
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


is_local = False


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument("--jobname", type=str)
    return parser.parse_args()


def configure_agent(params, env):
    agent = FQFAgent(env=env, test_env=env, params=params)
    return agent


def configure_behavior_space(params):
    return BehaviorSpace(params)


def configure_params(params, seed=None):
    import uuid
    experiment_seed = seed or str(uuid.uuid4())
    params["Experiment"]["random_seed"] = experiment_seed
    params["Experiment"]["dir"] = str(Path.home().joinpath("output/experiments/exp_{}".format(experiment_seed)))
    params["Experiment"]["params"] = "params_{}_{}.json"
    params["Experiment"]["scenarios_generated"] = "scenarios_list_{}_{}"
    params["Experiment"]["num_episodes"] = 50000
    params["Experiment"]["num_scenarios"] = 1000
    params["Experiment"]["map_filename"] = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
    return params



def configure_scenario_generation(num_scenarios, params):
    scenario_generation = ConfigurableScenarioGeneration(num_scenarios=num_scenarios,
                                                         params=params)
    return scenario_generation


def run(params, env, exp_exists=False):
    agent = configure_agent(params, env)
    if exp_exists:
      agent_checkpoint_last = os.path.join(params["Experiment"]["dir"], "agent/checkpoints/best")
      if os.path.isdir(agent_checkpoint_last):
        print("Loading from last best checkpoint.")
        agent.load_models(agent_checkpoint_last)
      else:
        print("No checkpoint written.")
    exp = Experiment(params=params, agent=agent)
    exp.run()


def check_if_exp_exists(params):
  return os.path.isdir(params["Experiment"]["dir"])


def main():
    args = configure_args()
    if is_local:
        dir_prefix = ""
    else:
        dir_prefix = "hy-x-beliefs.runfiles/hythe/"
    print("Executing job :", args.jobname)
    print("Experiment server at :", os.getcwd())
    params = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/fqf_params_default.json"),
                             log_if_default=True)
    params = configure_params(params, seed=args.jobname)
    experiment_id = params["Experiment"]["random_seed"]
    params_filename = os.path.join(params["Experiment"]["dir"], "params_{}.json".format(experiment_id))
    params_behavior_filename = os.path.join(params["Experiment"]["dir"], "behavior_params_{}.json".format(experiment_id))

    # check if exp exists and handle preemption
    exp_exists = check_if_exp_exists(params)
    if exp_exists:
      print("Loading existing experiment from: {}".format(args.jobname, (params["Experiment"]["dir"])))
      if os.path.isfile(params_filename):
        params = ParameterServer(filename=params_filename, log_if_default=True)
      if os.path.isfile(params_behavior_filename):
        params_behavior = ParameterServer(filename=params_behavior_filename, log_if_default=True)
    else:
      Path(params["Experiment"]["dir"]).mkdir(parents=True, exist_ok=True)
    params["ML"]["BaseAgent"]["SummaryPath"] = os.path.join(params["Experiment"]["dir"], "agent/summaries")
    params["ML"]["BaseAgent"]["CheckpointPath"] = os.path.join(params["Experiment"]["dir"], "agent/checkpoints")

    params_behavior = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/1D_desired_gap_no_prior.json"),
                                      log_if_default=True)
    params.Save(filename=params_filename)
    params_behavior.Save(filename=params_behavior_filename)

    # configure belief observer
    splits = 8
    behavior_space = configure_behavior_space(params_behavior)

    hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
    observer = BeliefObserver(params, hypothesis_set, splits=splits)
    behavior = BehaviorDiscreteMacroActionsML(params_behavior)
    evaluator = GoalReached(params)

    viewer = MPViewer(params=params,
                      x_range=[-35, 35],
                      y_range=[-35, 35],
                      follow_agent_id=True)


    # database creation
    dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=2,
                             num_serialize_scenarios=1000)
    dbs.process(os.path.join(dir_prefix, "configuration/database"), filter_sets="**/**/interaction_merging_light_dense_1D.json")
    local_release_filename = dbs.release(version="test")
    db = BenchmarkDatabase(database_root=local_release_filename)
    scenario_generator, _, _ = db.get_scenario_generator(scenario_set_id=0)
    env = HyDiscreteHighway(params=params,
                            scenario_generation=scenario_generator,
                            behavior=behavior,
                            evaluator=evaluator,
                            observer=observer,
                            viewer=viewer,
                            render=is_local)

    run(params, env, exp_exists)
    params.Save(filename=params_filename)
    logging.info('-' * 60)
    logging.info("Writing params to :{}".format(params_filename))
    logging.info('-' * 60)
    params_behavior.Save(filename=params_behavior_filename)
    logging.info('-' * 60)
    logging.info("Writing behavior params to :{}".format(params_behavior_filename))
    logging.info('-' * 60)

    return


if __name__ == '__main__':
    main()