from argparse import ArgumentParser
from datetime import datetime
import logging
import os
from pathlib import Path
from sys import argv
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy
from bark.core.models.dynamic import StateDefinition

from hythe.libs.experiments.experiment import Experiment
from bark.runtime.viewer.matplotlib_viewer import MPViewer

from bark.runtime.commons.parameters import ParameterServer

from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
  import LinearAnneaer
from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import IQNAgent
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.demonstrations import (
  DemonstrationCollector, DemonstrationGenerator)

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


is_local = True

if is_local:
  num_episodes = 5
  num_scenarios = 1
  trained_only_demonstrations_path = "/home/ekumar/output/experiments/exp_8b5b0dfc-3320-4033-8a7d-9446a60061cf"
else:
  num_episodes = 50000
  num_scenarios = 1000
  trained_only_demonstrations_path = "/mnt/glusterdata/home/ekumar/output/experiments/exp_stats_iqn_red_q/"


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
    ego_world_states = []
    memory = agent.memory
    learned_states = memory["state"]
    actions = memory['action']
    is_demos = memory['is_demo']
    for state, action, is_demo in zip(learned_states, actions, is_demos):
        ego_state = np.zeros((env._observer._len_ego_state + 1))
        ego_nn_input_state = deepcopy(state[0:env._observer._len_ego_state])
        ego_state[1:] = ego_nn_input_state
        reverted_observed_state = env._observer.rev_observe_for_ego_vehicle(ego_state)
        ego_world_states.append((reverted_observed_state[int(StateDefinition.X_POSITION)],
        reverted_observed_state[int(StateDefinition.Y_POSITION)],
        reverted_observed_state[int(StateDefinition.THETA_POSITION)],
        reverted_observed_state[int(StateDefinition.VEL_POSITION)], action[0], is_demo[0]))
    df = pd.DataFrame(ego_world_states, columns=['pos_x', 'pos_y', 'orientation', 'velocity', 'action', 'is_demo'])
    print(df.tail(10))
    if not os.path.exists(os.path.join(params["Experiment"]["dir"], "demonstrations")):
      os.makedirs(os.path.join(params["Experiment"]["dir"], "demonstrations"))
    df.to_pickle(os.path.join(params["Experiment"]["dir"], "demonstrations/learned_dataframe"))
    print("Training on", env._observer._world_x_range, env._observer._world_y_range)

def check_if_exp_exists(params):
  return os.path.isdir(params["Experiment"]["dir"])


def main():
    args = configure_args()
    if is_local:
        dir_prefix = ""
    else:
        dir_prefix="hy-iqn-lfd-exp.runfiles/hythe/"
    print("Executing job :", args.jobname)
    print("Experiment server at :", os.getcwd())
    params = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/iqn_params_demo.json"),
                             log_if_default=True)
    params = configure_params(params, seed=args.jobname)
    print(params["Experiment"]["num_episodes"])
    experiment_id = params["Experiment"]["random_seed"]
    params_filename = os.path.join(params["Experiment"]["dir"], "params_{}.json".format(experiment_id))

    behavior = BehaviorDiscreteMacroActionsML(params)
    evaluator = GoalReached(params)
    observer = NearestAgentsObserver(params)
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