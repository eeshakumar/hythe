import numpy as np
import os
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


class HyBeliefObserverationSpaceTEsts(unittest.TestCase):

    @timer
    def test_belief_observer_observation_space(self):
        params = ParameterServer(
            filename="configuration/params/fqf_params.json")
        splits = 8
        params_behavior = ParameterServer(
            filename="configuration/params/1D_desired_gap_no_prior.json")
        behavior_space = BehaviorSpace(params_behavior)
        hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
        observer = BeliefObserver(params, hypothesis_set, splits=splits)
        max_num_agents = observer._max_num_vehicles
        assert max_num_agents * len(hypothesis_set) == observer.max_beliefs


if __name__ == '__main__':
    unittest.main()