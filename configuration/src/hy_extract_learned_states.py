try:
  import debug_settings
except:
  pass

import os
import numpy as np
import pandas as pd
from copy import deepcopy

from bark.core.models.behavior import *
from bark.core.models.dynamic import StateDefinition

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.commons.parameters import ParameterServer
from hythe.libs.observer.belief_observer import BeliefObserver
from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime

from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.evaluators.goal_reached import GoalReached
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace

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

is_belief_observer = True
exp_root = "/home/ekumar/master_thesis/results/training/december/iqn/lfd/feb/exp_iqn_pre_exp_2sdi64"


def pick_agent(exp_root, env, params):
    agent_save_dir = os.path.join(exp_root, "agent")
    agent = IQNAgent(env=env, params=params, agent_save_dir=agent_save_dir,
                     checkpoint_load="trained_only_demonstrations", is_online_demo=True)
    print(agent.memory.capacity, agent.memory.agent_capacity, agent.memory._an, agent.memory._dn)
    return agent


def main():
    map_filename = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
    params = ParameterServer(filename=os.path.join(exp_root, "params_iqn_pre_exp_2sdi64.json"), log_if_default=True)
    params_behavior = ParameterServer(filename=os.path.join(exp_root, "behavior_params_iqn_pre_exp_2sdi64.json"))
    behavior = BehaviorDiscreteMacroActionsML(params)
    evaluator = GoalReached(params)
    if is_belief_observer:
        splits = 2
        behavior_space = BehaviorSpace(params_behavior)

        hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
        observer = BeliefObserver(params, hypothesis_set, splits=splits)
    else:
        observer = NearestAgentsObserver(params)
    viewer = MPViewer(params=params,
                    x_range=[-35, 35],
                    y_range=[-35, 35],
                    follow_agent_id=True)

    # database creation
    dbs1 = DatabaseSerializer(test_scenarios=1, test_world_steps=2,
                             num_serialize_scenarios=1)
    dbs1.process(os.path.join("", "configuration/database"),
      filter_sets="**/**/interaction_merging_light_dense_1D.json")
    local_release_filename = dbs1.release(version="lfd_offline")
    db1 = BenchmarkDatabase(database_root=local_release_filename)
    scenario_generator1, _, _ = db1.get_scenario_generator(0)

    env = HyDiscreteHighway(params=params,
                            scenario_generation=scenario_generator1,
                            map_filename=map_filename,
                            behavior=behavior,
                            evaluator=evaluator,
                            observer=observer,
                            viewer=viewer,
                            render=False)

    scenario, _ = scenario_generator1.get_next_scenario()
    world = scenario.GetWorldState()
    observer.Reset(world)
    env.reset()


    agent = pick_agent(exp_root, env, params)
    ego_world_states = []
    memory = agent.memory
    learned_states = memory["state"]
    actions = memory['action']
    is_demos = memory['is_demo']
    for state, action, is_demo in zip(learned_states, actions, is_demos):
        ego_state = np.zeros((observer._len_ego_state + 1))
        ego_nn_input_state = deepcopy(state[0:observer._len_ego_state])
        ego_state[1:] = ego_nn_input_state
        reverted_observed_state = observer.rev_observe_for_ego_vehicle(ego_state)
        ego_world_states.append((reverted_observed_state[int(StateDefinition.X_POSITION)],
        reverted_observed_state[int(StateDefinition.Y_POSITION)],
        reverted_observed_state[int(StateDefinition.THETA_POSITION)],
        reverted_observed_state[int(StateDefinition.VEL_POSITION)], action[0], is_demo[0]))
    df = pd.DataFrame(ego_world_states, columns=['pos_x', 'pos_y', 'orientation', 'velocity', 'action', 'is_demo'])
    print(df.head(10))
    if not os.path.exists(os.path.join(exp_root, "demonstrations")):
      os.makedirs(os.path.join(exp_root, "demonstrations"))
    df.to_pickle(os.path.join(exp_root, "demonstrations", "learned_dataframe"))
    return


if __name__ == "__main__":
    main()