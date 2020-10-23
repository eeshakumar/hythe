import time
import unittest
import numpy as np

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.memory import LazyPrioritizedDemMultiStepMemory


class TestBufferAccuracy(unittest.TestCase):

    def test(self):
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
        state = env.reset()
        print(state.shape)
        # env.reset()
        actions = [5] * 10
        print(actions)
        # state_shape = state.shape
        memory  = LazyPrioritizedDemMultiStepMemory(
            10,
            env.observation_space.shape,
            0,
            0.99,
            1,
            beta_steps=10,
            epsilon_demo=1.0,
            epsilon_alpha=0.001,
            per_beta=0.6,
            per_beta_steps=25,
            demo_ratio=0.25
        )
        print("MEM CAPACITY DEMO/AGENT", memory.demo_capacity, memory.agent_capacity)
        print(memory._dn, memory._an)
        for i, action in enumerate(actions):
            next_state, reward, done, _ = env.step(action)
            # print(next_state.shape)
            # print(reward)
            # print(done)
            if i%4 == 0:
                memory.append(state, action, reward, next_state, done, True)
            else:
                memory.append(state, action, reward, next_state, done, False)
            print("MEMORY", memory._n, memory._dn, memory._an)
            state = next_state
            time.sleep(0.2)
            memory.per_beta.step()
        batch_size = 2
        N = 2
        errors = np.ones((batch_size, N, N)) * 0.5
        # errors[0,:,:] = 200
        import torch
        errors = torch.from_numpy(errors)

        for i in range(10):
            print("SAMPLING!!!")
            (states, actions, rewards, next_states, dones, is_demos), weights = \
                memory.sample(batch_size)
            print("ST", state)
            print("NST", next_state)
            print(states.shape, actions, rewards, next_states.shape, dones, is_demos)
            memory.update_priority(errors, is_demos)


if __name__ == "__main__":
    unittest.main()