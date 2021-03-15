try:
  import debug_settings
except:
  pass

from argparse import ArgumentParser
from datetime import datetime
import logging
import os
from pathlib import Path
from sys import argv
import yaml
import numpy as np
from copy import deepcopy
import pandas as pd
from bark.core.models.dynamic import StateDefinition

from bark.core.models.behavior import *

from hythe.libs.experiments.experiment import Experiment
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from hythe.libs.observer.belief_observer import BeliefObserver

from bark.runtime.commons.parameters import ParameterServer

from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime

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
is_generate_demonstrations = True
is_train_on_demonstrations = False

capacity = 100000

if is_local:
  num_episodes = 100
  num_scenarios = 10000
  num_demo_episodes = num_scenarios
  demo_dir_default = None
else:
  num_episodes = 50000
  num_scenarios = 100
  num_demo_episodes = 1000
  # default dir with demonstrations data
  demo_dir_default = "/mnt/glusterdata/home/ekumar/demonstrations/exp_aa3feb5b-f98e-4318-9bb8-c183eaad4c3a"


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument("--jobname", type=str)
    parser.add_argument("--grad_update_steps", type=int, default=500)
    parser.add_argument("--demodir", type=str, default=demo_dir_default)
    return parser.parse_args()


def configure_agent(params, env):
    agent_save_dir = os.path.join(params["Experiment"]["dir"], "agent")
    training_benchmark = None #TrainingBenchmarkDatabase()
    agent = IQNAgent(env=env, params=params, agent_save_dir=agent_save_dir,
                     training_benchmark=training_benchmark, is_learn_from_demonstrations=True)
    return agent


def generate_demonstrations(params, env, eval_criteria, demo_behavior=None, use_mp_runner=True, db=None):
    demo_collector = DemonstrationCollector()
    save_dir = os.path.join(params["Experiment"]["dir"], "demonstrations")
    demo_generator = DemonstrationGenerator(env, params, demo_behavior, demo_collector, save_dir)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    demo_generator.generate_demonstrations(num_demo_episodes, eval_criteria, save_dir, use_mp_runner=use_mp_runner, db=db)
    demo_generator.dump_demonstrations(save_dir)
    return demo_generator.demonstrations


def generate_uct_hypothesis_behavior():
    ml_params = ParameterServer(filename="configuration/params/iqn_params_demo_full.json", log_if_default=True)
    behavior_ml_params = ml_params["ML"]["BehaviorMPMacroActions"]
    mcts_params = ParameterServer(filename="configuration/params/default_uct_params.json", log_if_default=True)
    mcts_params["BehaviorUctBase"]["EgoBehavior"] = behavior_ml_params
    behavior = BehaviorUCTHypothesis(mcts_params, [])
    mcts_params.Save(filename="./default_uct_params.json")
    return behavior, mcts_params


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


def unpack_load_demonstrations(demo_root):
    demo_dir = os.path.join(demo_root, "demonstrations/generated_demonstrations")
    collector = DemonstrationCollector.load(demo_dir)
    return collector, collector.GetDemonstrationExperiences()


def run(args, params, env, exp_exists=False, db=None):

    demonstrations = None
    # add an eval criteria and generate demonstrations
    if is_generate_demonstrations:
      eval_criteria = {"goal_reached" : lambda x : x}
      demo_behavior, mcts_params = generate_uct_hypothesis_behavior()
      demonstrations = generate_demonstrations(params, env, eval_criteria, demo_behavior, db=db)
      print("Total demonstrations generated", len(demonstrations))
      ego_world_states = []
      # _, demonstrations = unpack_load_demonstrations(demo_root)
      for demo in demonstrations:
          ego_state = np.zeros((env._observer._len_ego_state + 1))
          (nn_ip_state, action, reward, next_state, done, is_demo) = demo
          ego_nn_input_state = deepcopy(nn_ip_state[0:env._observer._len_ego_state])
          ego_state[1:] = ego_nn_input_state
          reverted_observed_state = env._observer.rev_observe_for_ego_vehicle(ego_state)
          ego_world_states.append((reverted_observed_state[int(StateDefinition.X_POSITION)], reverted_observed_state[int(StateDefinition.Y_POSITION)], reverted_observed_state[int(StateDefinition.THETA_POSITION)], reverted_observed_state[int(StateDefinition.VEL_POSITION)], action, int(is_demo)))  
      df = pd.DataFrame(ego_world_states, columns=['pos_x', 'pos_y', 'orientation', 'velocity', 'action', 'is_demo'])
      print(df.head(10))
      df.to_pickle(os.path.join(params["Experiment"]["dir"], "demonstrations/demo_dataframe"))
      print("Picked demo unormed to ", os.path.join(params["Experiment"]["dir"], "demonstrations/demo_dataframe"))
      print("Training off", env._observer._world_x_range, env._observer._world_y_range)
    if is_train_on_demonstrations:
      if args.demodir:
        collector, demonstrations = unpack_load_demonstrations(args.demodir)
      else:
        collector, demonstrations = unpack_load_demonstrations(params["Experiment"]["dir"])
      if params["ML"]["BaseAgent"]["Multi_step"] is not None:
        multistep_capacity = capacity + params["ML"]["BaseAgent"]["Multi_step"] - 1 
      if multistep_capacity < len(demonstrations):
        demonstrations = demonstrations[-multistep_capacity:]
        print("Pruned number of demonstrations", len(demonstrations))
      else:
        print("Number of demonstrations under capacity requested, using full demonstrations")
      print("Loaded number of demonstrations", len(demonstrations))
      if is_local:
        # Assign steps by args
        params["ML"]["DemonstratorAgent"]["Agent"]["online_gradient_update_steps"] = args.grad_update_steps
      # Assign capacity by length of demonstrations
      params["ML"]["BaseAgent"]["MemorySize"] = len(demonstrations)
      params["ML"]["DemonstratorAgent"]["Buffer"]["demo_ratio"] = 1.0
      agent = configure_agent(params, env)
      exp = Experiment(params=params, agent=agent, dump_scenario_interval=25000)
      exp.run(demonstrator=True, demonstrations=demonstrations, num_episodes=num_episodes, learn_only=True)
      print("Training off", env._observer._world_x_range, env._observer._world_y_range)


def check_if_exp_exists(params):
  return os.path.isdir(params["Experiment"]["dir"])


def main():
    args = configure_args()
    if is_local:
        dir_prefix = ""
    else:
        dir_prefix="hy-iqnfd-exp.runfiles/hythe/"
    print("Executing job :", args.jobname)
    print("Experiment server at :", os.getcwd())
    params = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/iqn_params_demo_full_local.json"),
                             log_if_default=True)
    params = configure_params(params, seed=args.jobname)
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
    dbs = DatabaseSerializer(test_scenarios=1, test_world_steps=2,
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
    scenario, _ = scenario_generator.get_next_scenario()
    world = scenario.GetWorldState()
    observer.Reset(world)

    assert env.action_space._n == 8, "Action Space is incorrect!"
    run(args, params, env, db=db)
    params.Save(params_filename)
    logging.info('-' * 60)
    logging.info("Writing params to :{}".format(params_filename))
    logging.info('-' * 60)


if __name__ == '__main__':
    main()