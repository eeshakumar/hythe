configuration/src/BUILDimport os
import sys
import logging
import glob
# import matplotlib.pyplot as plt
from argparse import ArgumentParser

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP, BenchmarkRunner
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from hythe.libs.observer.belief_observer import BeliefObserver
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver


from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.viewer import generatePoseFromState
from bark.runtime.viewer.video_renderer import VideoRenderer

import bark.core
import bark.core.models.behavior
from bark.core.models.dynamic import StateDefinition
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import IQNAgent
from hythe.libs.evaluation.training_benchmark_database import TrainingBenchmarkDatabase
from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime
from hythe.libs.experiments.experiment import Experiment

from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


is_belief_observer = False

is_local = True

if is_local:
  # num_episodes = 100
  num_scenarios = 2
else:
  # num_episodes = 50000
  num_scenarios = 1000

def configure_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_dir', "--edir", type=str)
    parser.add_argument("--num_episodes", '--epi', default=75000, type=int)
    return parser.parse_args(sys.argv[1:])


def load_params(exp_dir):
    params_filename = glob.glob(os.path.join(exp_dir, "params*"))[0]
    return params_filename, ParameterServer(filename=params_filename, log_if_default=True)


def load_agent(params, env, exp_dir, checkpoint='last'):
    agent_dir = os.path.join(exp_dir, 'agent')
    return IQNAgent(params=params, env=env, agent_save_dir=agent_dir, checkpoint_load=checkpoint)


def resume_experiment(params, num_episodes, agent):
    exp = Experiment(params=params, agent=agent)
    exp.resume(num_episodes)

args = configure_args()
exp_dir = args.exp_dir
num_episodes = args.num_episodes

params_filename, params = load_params(exp_dir)

dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=20, num_serialize_scenarios=num_scenarios)
dbs.process("configuration/database", filter_sets="**/**/interaction_merging_light_dense_1D.json")
local_release_filename = dbs.release(version="resume")

db = BenchmarkDatabase(database_root=local_release_filename)
scenario_generator, _, _ = db.get_scenario_generator(0)

# load belief observer specifics
if is_belief_observer:
  splits = 2
  behavior_params_filename = glob.glob(os.path.join(exp_dir, "behavior_params*"))[0]
  params_behavior = ParameterServer(filename=behavior_params_filename, log_if_default=True)
  behavior_space = BehaviorSpace(params_behavior)

  hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
  observer = BeliefObserver(params, hypothesis_set, splits=splits)
  behavior = BehaviorDiscreteMacroActionsML(params_behavior)
# if not, load default observer
else:
  behavior = BehaviorDiscreteMacroActionsML(params)
  observer = NearestAgentsObserver(params)

evaluator = GoalReached(params)

viewer = MPViewer(
  params=params,
  center= [960, 1000.8],
  enforce_x_length=True,
  x_length = 100.0,
  use_world_bounds=False)

# load env
env = HyDiscreteHighway(params=params,
                        scenario_generation=scenario_generator,
                        behavior=behavior,
                        evaluator=evaluator,
                        observer=observer,
                        viewer=viewer,
                        render=False)

agent = load_agent(params, env, exp_dir, 'last')
resume_experiment(params=params, num_episodes=num_episodes, agent=agent)

params.Save(filename=params_filename)
logging.info('-' * 60)
logging.info("Writing params to :{}".format(params_filename))
logging.info('-' * 60)