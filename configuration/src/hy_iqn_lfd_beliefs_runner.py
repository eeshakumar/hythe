from argparse import ArgumentParser
from datetime import datetime
import logging
import os
from pathlib import Path
from sys import argv
import yaml

from hythe.libs.experiments.experiment import Experiment
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from hythe.libs.observer.belief_observer import BeliefObserver

from bark.runtime.commons.parameters import ParameterServer

from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
  import LinearAnneaer
from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import IQNAgent
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.demonstrations import (
  DemonstrationCollector, DemonstrationGenerator)

from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace

from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

from libs.evaluation.training_benchmark_database import TrainingBenchmarkDatabase

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.\
            tests.test_demo_behavior import TestDemoBehavior          


is_local = False

if is_local:
  num_episodes = 10
  num_scenarios = 1
  trained_only_demonstrations_path = "/home/ekumar/master_thesis/results/training/december/iqn/lfd/stats/exp_2s_be_di64"
else:
  num_episodes = 50000
  num_scenarios = 1000
  trained_only_demonstrations_path = "/mnt/glusterdata/home/ekumar/output/experiments/exp_2s_be_di64"


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument("--jobname", type=str)
    return parser.parse_args()


def configure_agent(params, env):
    agent_save_dir = os.path.join(trained_only_demonstrations_path, "agent")
    training_benchmark = None #TrainingBenchmarkDatabase()
    agent = IQNAgent(env=env, params=params, agent_save_dir=agent_save_dir,
                     training_benchmark=training_benchmark, is_learn_from_demonstrations=True,
                     checkpoint_load="trained_only_demonstrations", is_online_demo=True)
    return agent


def configure_params(params, seed=None):
    import uuid
    experiment_seed = seed or str(uuid.uuid4())
    params["Experiment"]["random_seed"] = experiment_seed
    params["Experiment"]["dir"] = str(Path.home().joinpath("output/experiments/exp_{}".format(experiment_seed)))
    Path(params["Experiment"]["dir"]).mkdir(parents=True, exist_ok=True)
    params["Experiment"]["params"] = "params_{}_{}.json"
    params["Experiment"]["scenarios_generated"] = "scenarios_list_{}_{}"
    params["Experiment"]["num_episodes"] = num_episodes
    params["Experiment"]["num_scenarios"] = num_scenarios
    params["Experiment"]["map_filename"] = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
    return params


def run(params, env, exp_exists=False):
    agent = configure_agent(params, env)
    logging.info(f"target update every: {agent.target_update_interval}")
    logging.info(f"summary written every: {agent.summary_log_interval}")
    exp = Experiment(params=params, agent=agent, dump_scenario_interval=25000)
    exp.run(demonstrator=True, demonstrations=None, num_episodes=num_episodes)


def check_if_exp_exists(params):
  return os.path.isdir(params["Experiment"]["dir"])


def main():
    args = configure_args()
    if is_local:
        dir_prefix = ""
    else:
        dir_prefix="hy-iqn-lfd-beliefs-exp.runfiles/hythe/"
    print("Executing job :", args.jobname)
    print("Experiment server at :", os.getcwd())
    print(dir_prefix)
    params = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/iqn_params_demo.json"),
                             log_if_default=True)
    params = configure_params(params, seed=args.jobname)
    print(params["Experiment"]["num_episodes"])
    experiment_id = params["Experiment"]["random_seed"]
    params_filename = os.path.join(params["Experiment"]["dir"], "params_{}.json".format(experiment_id))

    # configure belief observer 
    params_behavior_filename = os.path.join(params["Experiment"]["dir"], "behavior_params_{}.json".format(experiment_id))
    params_behavior = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/1D_desired_gap_no_prior.json"),
                                      log_if_default=True)
    params_behavior.Save(filename=params_behavior_filename)

    splits = 2
    behavior_space = BehaviorSpace(params_behavior)

    hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
    observer = BeliefObserver(params, hypothesis_set, splits=splits)
  
    behavior = BehaviorDiscreteMacroActionsML(params)
    evaluator = GoalReached(params)
    viewer = MPViewer(params=params,
                      x_range=[-35, 35],
                      y_range=[-35, 35],
                      follow_agent_id=True)

    # extract params and save experiment parameters
    params["ML"]["BaseAgent"]["SummaryPath"] = os.path.join(params["Experiment"]["dir"], "agent/summaries")
    params["ML"]["BaseAgent"]["CheckpointPath"] = os.path.join(params["Experiment"]["dir"], "agent/checkpoints")

    params.Save(filename=params_filename)
    logging.info('-' * 60)
    logging.info("Writing params to :{}".format(params_filename))
    logging.info('-' * 60)

    # database creation
    dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=2,
                             num_serialize_scenarios=num_scenarios)
    dbs.process(os.path.join(dir_prefix, "configuration/database"), filter_sets="**/**/interaction_merging_light_dense_1D.json")
    local_release_filename = dbs.release(version="test")
    db = BenchmarkDatabase(database_root=local_release_filename)
    scenario_generator, _, _ = db.get_scenario_generator(0)

    env = HyDiscreteHighway(params=params,
                            scenario_generation=scenario_generator,
                            behavior=behavior,
                            evaluator=evaluator,
                            observer=observer,
                            viewer=viewer,
                            render=False)
    assert env.action_space._n == 8, "Action Space is incorrect!"
    run(params, env)
    params.Save(params_filename)
    logging.info('-' * 60)
    logging.info("Writing params to :{}".format(params_filename))
    logging.info('-' * 60)


if __name__ == '__main__':
    main()