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
from copy import deepcopy
import time
import unittest

from bark.core.models.behavior import *

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

demo_root = "/home/ekumar/output/experiments/exp_72f85489-b3db-4fdf-99fe-fe48c2b0730e"

def unpack_load_demonstrations(demo_root):
    demo_dir = os.path.join(demo_root, "demonstrations/generated_demonstrations")
    collector = DemonstrationCollector.load(demo_dir)
    return collector, collector.GetDemonstrationExperiences()


class TestObsTrajExtraction(unittest.TestCase):


    def test_obs_traj(self):
        map_filename = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
        params = ParameterServer()
        behavior = BehaviorDiscreteMacroActionsML(params)
        evaluator = GoalReached(params)
        observer = NearestAgentsObserver(params)
        viewer = MPViewer(params=params,
                        x_range=[-35, 35],
                        y_range=[-35, 35],
                        follow_agent_id=True)
        env = HyDiscreteHighway(params=params,
                                map_filename=map_filename,
                                behavior=behavior,
                                evaluator=evaluator,
                                observer=observer,
                                viewer=viewer,
                                render=True)
        env.reset()
        import numpy as np
        actions = np.random.randint(0, 7, 100)
        for action in actions:

            concatenated_state, _, _, _ = env.step(action)
            nn_ip_state = concatenated_state
            ego_nn_input_state = deepcopy(concatenated_state[0:observer._len_ego_state])
            reverted_observed_state = observer.rev_observe_for_ego_vehicle(nn_ip_state)
            ext_reverted_observed_state = np.zeros((reverted_observed_state.shape[0] + 1))
            ext_reverted_observed_state[1:] = reverted_observed_state
            renormed_ego_state = observer._select_state_by_index(observer._norm(ext_reverted_observed_state))
            time.sleep(0.2)
            np.testing.assert_array_almost_equal(ego_nn_input_state, renormed_ego_state)


if __name__ == '__main__':
    unittest.main()