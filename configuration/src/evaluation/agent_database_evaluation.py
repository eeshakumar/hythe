try:
    import debug_settings
except:
    pass

import os
import sys
import logging
import glob
# import matplotlib.pyplot as plt
from argparse import ArgumentParser

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info("Running on process with ID: {}".format(os.getpid()))
import bark.core.commons

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP, BenchmarkRunner
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from hythe.libs.observer.belief_observer import BeliefObserver
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime


from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.viewer import generatePoseFromState
from bark.runtime.viewer.video_renderer import VideoRenderer

import bark.core
import bark.core.models.behavior
from bark.core.models.dynamic import StateDefinition
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent


from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

log_folder = os.path.abspath(os.path.join(os.getcwd(), "logs"))
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
logging.info("Logging into: {}".format(log_folder))
bark.core.commons.GLogInit(sys.argv[0], log_folder, 3, True)

# reduced max steps and scenarios for testing
max_steps = 50
num_scenarios = 10

def configure_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', "--ckpd", type=str)
    return parser.parse_args(sys.argv[1:])

logging.getLogger().setLevel(logging.INFO)

print("Experiment server at :", os.getcwd())

dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=20, num_serialize_scenarios=num_scenarios)
dbs.process("configuration/database", filter_sets="**/**/interaction_merging_light_dense_1D.json")
local_release_filename = dbs.release(version="tmp2")

db = BenchmarkDatabase(database_root=local_release_filename)
scenario_generator, _, _ = db.get_scenario_generator(0)

evaluators = {"success" : "EvaluatorGoalReached", "collision_other" : "EvaluatorCollisionEgoAgent",
       "out_of_drivable" : "EvaluatorDrivableArea", "max_steps": "EvaluatorStepCount"}
terminal_when = {"collision_other" : lambda x: x, "out_of_drivable" : lambda x: x, "max_steps": lambda x : x>max_steps, "success" : lambda x : x}

args = configure_args()
exp_dir = args.checkpoint_dir
if exp_dir is None:
  exp_dir = "results/training/toy_benchmark"

params_filename = glob.glob(os.path.join(exp_dir, "params*"))[0]
params = ParameterServer(filename=params_filename, log_if_default=True)
params["ML"]["BaseAgent"]["SummaryPath"] = os.path.join(exp_dir, "agent/summaries")
params["ML"]["BaseAgent"]["CheckpointPath"] = os.path.join(exp_dir, "agent/checkpoints")

# create env
splits = 8
behavior_params_filename = glob.glob(os.path.join(exp_dir, "behavior_params*"))[0]
params_behavior = ParameterServer(filename=behavior_params_filename, log_if_default=True)
behavior_space = BehaviorSpace(params_behavior)

hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
observer = BeliefObserver(params, hypothesis_set, splits=splits)
behavior = BehaviorDiscreteMacroActionsML(params_behavior)

env_to_pass_observer_behavior = SingleAgentRuntime(ml_behavior=behavior,
                                                  observer=observer,
                                                  step_time=-1.0,
                                                  viewer=-1.0,
                                                  scenario_generator=-1.0,
                                                  evaluator=-1.0)

# load agent
agent = FQFAgent(env=env_to_pass_observer_behavior, test_env=None, params=params)
agent.load_models(os.path.join(exp_dir, "agent/checkpoints/best"))

behaviors = {"behavior_fqf_agent": agent}
benchmark_runner = BenchmarkRunner(benchmark_database = db,
                                    evaluators = evaluators,
                                    terminal_when = terminal_when,
                                    behaviors = behaviors,
                                    num_scenarios=num_scenarios,
                                    log_eval_avg_every = 10,
                                    checkpoint_dir = "checkpoints",
                                    deepcopy=False)

viewer = MPViewer(
  params=params,
  center= [960, 1000.8],
  enforce_x_length=True,
  x_length = 100.0,
  use_world_bounds=False)
viewer.show()
result = benchmark_runner.run(viewer=viewer)

print(result.get_data_frame())
result.dump(os.path.join("./benchmark_results"))

result_loaded = result.load(os.path.join("./benchmark_results"))
print(result.get_data_frame())