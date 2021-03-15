import time
import unittest

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime


class TestHyEnv(unittest.TestCase):

    def test_env(self):
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
        actions = [5]*100
        print(actions)
        for action in actions:
            env.step(action)
            time.sleep(0.2)
        return

if __name__ == '__main__':
    unittest.main()