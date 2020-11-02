try:
    import debug_settings
except:
    pass

import os
import sys
import logging
import matplotlib.pyplot as plt
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info("Running on process with ID: {}".format(os.getpid()))
import bark.core.commons

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP, BenchmarkRunner

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.viewer import generatePoseFromState
from bark.runtime.viewer.video_renderer import VideoRenderer

from bark.core.models.dynamic import StateDefinition
from bark.core.models.behavior import BehaviorMobil


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

logging.getLogger().setLevel(logging.INFO)


dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=20, num_serialize_scenarios=num_scenarios)
dbs.process("configuration/database", filter_sets="**/**/interaction_merging_light_dense.json")
local_release_filename = dbs.release(version="tmp2")

db = BenchmarkDatabase(database_root=local_release_filename)

evaluators = {"success" : "EvaluatorGoalReached", "collision_other" : "EvaluatorCollisionEgoAgent",
       "out_of_drivable" : "EvaluatorDrivableArea", "max_steps": "EvaluatorStepCount"}
terminal_when = {"collision_other" : lambda x: x, "out_of_drivable" : lambda x: x, "max_steps": lambda x : x>max_steps, "success" : lambda x : x}

params = ParameterServer()
behaviors = {"behavior_ckpt1" : BehaviorMobil(params) }

benchmark_runner = BenchmarkRunner(benchmark_database = db,
                                    evaluators = evaluators,
                                    terminal_when = terminal_when,
                                    behaviors = behaviors, 
                                    num_scenarios=num_scenarios,
                                   log_eval_avg_every = 10,
                                   checkpoint_dir = "checkpoints",
                                   deepcopy=False)

viewer = MPViewer(
  params=ParameterServer(),
  center= [960, 1000.8],
  enforce_x_length=True,
  x_length = 100.0,
  use_world_bounds=False)
viewer.show()
result = benchmark_runner.run(maintain_history=True, viewer=viewer)

print(result.get_data_frame())
result.dump(os.path.join("./benchmark_results"), dump_histories=True, dump_configs=True)

result_loaded = result.load(os.path.join("./benchmark_results"), load_histories=True, load_configs=True)
print(result.get_data_frame())