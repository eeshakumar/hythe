import os
from copy import deepcopy
import numpy as np
import unittest
from bark.runtime.viewer.matplotlib_viewer import MPViewer

from bark.runtime.commons.parameters import ParameterServer
from hythe.libs.observer.belief_observer import BeliefObserver
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import ConfigurableScenarioGeneration
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.evaluators.goal_reached import GoalReached
from hythe.libs.environments.gym import HyDiscreteHighway
from hythe.libs.timer.timer import timer
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


class HyBeliefObserverTests(unittest.TestCase):

    def test_thresholding(self):
        params = ParameterServer()
        # init behavior space
        splits = 2
        behavior_params = ParameterServer()
        behavior_space = BehaviorSpace(behavior_params)
        hypothesis_set, _ = behavior_space.create_hypothesis_set_fixed_split(split=splits)
        observer = BeliefObserver(params, hypothesis_set, splits=splits)
        observer.is_enable_threshold = True
        
        belief_arr = np.linspace(0, 1, 100)
        belief_arr_copy = deepcopy(belief_arr)
        th_belief_arr = observer.threshold_beliefs(belief_arr)
        belief_arr_copy[belief_arr_copy <= observer.threshold] = 0.

        self.assertTrue(all(belief_arr_copy == th_belief_arr))


    def test_discretization(self):
        params = ParameterServer()
        # init behavior space
        splits = 2
        behavior_params = ParameterServer()
        behavior_space = BehaviorSpace(behavior_params)
        hypothesis_set, _ = behavior_space.create_hypothesis_set_fixed_split(split=splits)
        observer = BeliefObserver(params, hypothesis_set, splits=splits)
        observer.is_discretize = True
        
        belief_arr = np.linspace(0, 1, 100)
        belief_arr_copy = deepcopy(belief_arr)

        o_bucketized = observer.discretize_beliefs(belief_arr)

        bins_idxs = np.digitize(belief_arr_copy, observer.buckets, right=observer.is_ciel)
        self.assertEqual(bins_idxs.shape[0], belief_arr_copy.shape[0])
        bucketized = observer.buckets[bins_idxs]

        self.assertTrue(all(o_bucketized == bucketized))


if __name__ == '__main__':
    unittest.main()