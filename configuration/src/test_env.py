import unittest

from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML


class TestHyEnv(unittest.Testcase):

    def test_env(self):
        print("TESSSTtt")
        params = ParameterServer()
        behavior = BehaviorDiscreteMacroActionsML(params)
        evaluator = GoaleReached(params)
        observer = NearestAgentsObserver(params)

        return