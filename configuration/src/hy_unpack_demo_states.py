import os
import numpy as np
import pandas as pd
from copy import deepcopy
import glob
from bark.core.models.behavior import *
from bark.core.models.dynamic import StateDefinition

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.commons.parameters import ParameterServer

from hythe.libs.observer.belief_observer import BeliefObserver
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime

from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.demonstrations import (
  DemonstrationCollector, DemonstrationGenerator)

from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

is_belief_observer = True
demo_root = "/home/ekumar/output/experiments/exp_2474ef6b-d395-4410-9f6c-320028603322"

def unpack_load_demonstrations(demo_root):
    demo_dir = os.path.join(demo_root, "demonstrations/generated_demonstrations")
    collector = DemonstrationCollector.load(demo_dir)
    return collector, collector.GetDemonstrationExperiences()

def main():
    map_filename = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
    params_filename = glob.glob(os.path.join(demo_root, "params_[!behavior]*"))
    params = ParameterServer(filename=params_filename[0], log_if_default=True)
    behavior = BehaviorDiscreteMacroActionsML(params)
    evaluator = GoalReached(params)
    if is_belief_observer:
        splits = 2
        bparams_filename = glob.glob(os.path.join(demo_root, "behavior_*"))
        params_behavior = ParameterServer(filename=bparams_filename[0])
        behavior_space = BehaviorSpace(params_behavior)

        hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
        observer = BeliefObserver(params, hypothesis_set, splits=splits)
    else:
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
                            render=False)

    env.reset()

    ego_world_states = []
    _, demonstrations = unpack_load_demonstrations(demo_root)
    for demo in demonstrations:
        ego_state = np.zeros((observer._len_ego_state + 1))
        (nn_ip_state, action, reward, next_state, done, is_demo) = demo
        ego_nn_input_state = deepcopy(nn_ip_state[0:observer._len_ego_state])
        ego_state[1:] = ego_nn_input_state
        reverted_observed_state = observer.rev_observe_for_ego_vehicle(ego_state)
        ego_world_states.append((reverted_observed_state[int(StateDefinition.X_POSITION)],
        reverted_observed_state[int(StateDefinition.Y_POSITION)],
        reverted_observed_state[int(StateDefinition.THETA_POSITION)],
        reverted_observed_state[int(StateDefinition.VEL_POSITION)], action, int(is_demo)))
    df = pd.DataFrame(ego_world_states, columns=['pos_x', 'pos_y', 'orientation', 'velocity', 'action', 'is_demo'])
    print(df.head(10))
    df.to_pickle(os.path.join(demo_root, "demonstrations/demo_dataframe"))
    return

if __name__ == "__main__":
    main()