try:
    import debug_settings
except:
    pass

import os
import pickle
import sys
import logging
import glob
# import matplotlib.pyplot as plt
from argparse import ArgumentParser

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info("Running on process with ID: {}".format(os.getpid()))
import bark.core.commons

from bark.runtime.scenario import Scenario

from bark.benchmark.benchmark_result import BenchmarkResult
from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP, BenchmarkRunner
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from hythe.libs.observer.belief_observer import BeliefObserver
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver


from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.viewer import generatePoseFromState
from bark.runtime.viewer.video_renderer import VideoRenderer

import bark.core
import bark.core.models.behavior
from bark.core.models.dynamic import StateDefinition
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import IQNAgent
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.demonstrations import (
  DemonstrationCollector, DemonstrationGenerator)

from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

# Can be used when the demonstrator collector bugs out and hangs, but the checkpoints can be found 
# in the bazel exec.

def to_pickle(obj, dir, file):
  path = os.path.join(dir, file)
  with open(path, 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dump():
  ckpt_dir = [os.path.join('/home/ekumar/master_thesis/code/hythe-src/checkpoints/', ck) for ck in os.listdir("/home/ekumar/master_thesis/code/hythe-src/checkpoints/")]
  eval_criteria = {"goal_reached" : lambda x : x}
  print(ckpt_dir)
  demos_dir = os.path.join('/home/ekumar/demos/')
  list_of_demos = []
  for cdir in ckpt_dir:
    print(f"Extracting result {cdir}")
    result = BenchmarkResult.load_results(cdir)
    democ = DemonstrationCollector()
    democ._collection_result = result
    democ._directory = demos_dir
    demos = democ.ProcessCollectionResult(eval_criteria)
    list_of_demos.extend(demos)
  # make the demonstrations dir in the exp root
  os.makedirs("/home/ekumar/output/experiments/exp_c76fc949-e95f-4774-91ba-6bec575ada37/demonstrations/generated_demonstrations")
  to_pickle(list_of_demos, "/home/ekumar/output/experiments/exp_c76fc949-e95f-4774-91ba-6bec575ada37/demonstrations/generated_demonstrations", "demonstrations")
  collector = DemonstrationCollector.load( "/home/ekumar/output/experiments/exp_c76fc949-e95f-4774-91ba-6bec575ada37/demonstrations/generated_demonstrations")
  print("Total demonstations found:", len(collector.GetDemonstrationExperiences()))
  return

if __name__ == '__main__':
  dump()