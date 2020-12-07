from argparse import ArgumentParser
from datetime import datetime
import logging
import os
from pathlib import Path
from sys import argv
import yaml

from hythe.libs.experiments.experiment import Experiment
from bark.runtime.viewer.matplotlib_viewer import MPViewer

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

if is_local:
  num_episodes = 10
  num_scenarios = 6
  num_demo_episodes = 4
else:
  num_episodes = 50000
  num_scenarios = 1000
  num_demo_episodes = 1000


# TODO: Remove this evaluator
from bark_ml.evaluators.evaluator import StateEvaluator
class TestEvaluator(StateEvaluator):
  reach_goal = True
  def __init__(self,
              params=ParameterServer()):
    StateEvaluator.__init__(self, params)
    self.step = 0

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    reward = 0.0
    info = {"goal_r1" : False}
    if self.step > 2:
      done = True
      if self.reach_goal:
        reward = 0.1
        info = {"goal_r1" : True}
    self.step += 1
    return reward, done, info

  def Reset(self, world):
    self._step = 0
    #every second scenario goal is not reached
    TestEvaluator.reach_goal = not TestEvaluator.reach_goal


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument("--jobname", type=str)
    return parser.parse_args()


def configure_agent(params, env):
    agent_save_dir = os.path.join(params["Experiment"]["dir"], "agent")
    training_benchmark = None #TrainingBenchmarkDatabase()
    agent = IQNAgent(env=env, params=params, agent_save_dir=agent_save_dir,
                     training_benchmark=training_benchmark, is_learn_from_demonstrations=True)
    return agent


def generate_demonstrations(params, env, eval_criteria, demo_behavior=None):
    # TODO: Real objects need not be copied this way. Demo env can be created new.
    real_behavior = env._ml_behavior
    real_evaluator = env._evaluator
    env._evaluator = TestEvaluator()
    demo_behavior = TestDemoBehavior(params)
    demo_collector = DemonstrationCollector()
    save_dir = os.path.join(params["Experiment"]["dir"], "demonstrations")
    demo_generator = DemonstrationGenerator(env, params, demo_behavior, demo_collector, save_dir)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    demo_generator.generate_demonstrations(num_demo_episodes, eval_criteria, save_dir)
    demo_generator.dump_demonstrations(save_dir)
    env._evaluator = real_evaluator
    env._ml_behavior = real_behavior
    return demo_generator.demonstrations


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
    if exp_exists:
      agent_checkpoint_last = os.path.join(params["Experiment"]["dir"], "agent/checkpoints/best")
      if os.path.isdir(agent_checkpoint_last):
        print("Loading from last best checkpoint.")
        agent.load_models(agent_checkpoint_last)
      else:
        print("No checkpoint written.")

    #TODO: Revisit this criteria
    eval_criteria = {"goal_r1" : lambda x : x}
    demonstrations = generate_demonstrations(params, env, eval_criteria)
    exp = Experiment(params=params, agent=agent, dump_scenario_interval=25000)
    exp.run(demonstrator=True, demonstrations=demonstrations, learn_only=True)


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
    params = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/iqn_params.json"),
                             log_if_default=True)
    params = configure_params(params, seed=args.jobname)
    experiment_id = params["Experiment"]["random_seed"]
    params_filename = os.path.join(params["Experiment"]["dir"], "params_{}.json".format(experiment_id))

    # check if exp exists and handle preemption
    exp_exists = check_if_exp_exists(params)
    if exp_exists:
      print("Loading existing experiment from: {}".format(args.jobname, (params["Experiment"]["dir"])))
      if os.path.isfile(params_filename):
        params = ParameterServer(filename=params_filename, log_if_default=True)
    else:
      Path(params["Experiment"]["dir"]).mkdir(parents=True, exist_ok=True)  

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
                            render=True)
    assert env.action_space._n == 8, "Action Space is incorrect!"
    run(params, env, exp_exists)
    params.Save(params_filename)
    logging.info('-' * 60)
    logging.info("Writing params to :{}".format(params_filename))
    logging.info('-' * 60)


if __name__ == '__main__':
    main()