try:
  import debug_settings
except:
  pass

from argparse import ArgumentParser
from datetime import datetime
import logging
import os
import numpy as np
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


class TestObsTrajExtraction(unittest.TestCase):


    def test_obs_traj(self):
        map_filename = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
        params = ParameterServer()
        observer = NearestAgentsObserver(params)

        test_agent_state = np.asarray([0.0, 0.5, 0.5, 0.5, 0.5])
        copy_test_agent_state = deepcopy(test_agent_state)
        normed_state = observer._norm(copy_test_agent_state)
        copy_normed_state = deepcopy(normed_state)
        unnormed_state = observer._reverse_norm(copy_normed_state)
        np.testing.assert_array_almost_equal(test_agent_state, unnormed_state)


if __name__ == '__main__':
    unittest.main()