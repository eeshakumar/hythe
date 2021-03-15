try:
    import debug_settings
except:
    pass

import glob
import os
import sys
import logging
import glob
# import matplotlib.pyplot as plt
from argparse import ArgumentParser

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info("Running on process with ID: {}".format(os.getpid()))
import bark.core.commons

from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from hythe.libs.observer.belief_observer import BeliefObserver
from hythe.libs.environments.gym import HyDiscreteHighway
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver

from bark.runtime.scenario.scenario_generation import ConfigurableScenarioGeneration
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.viewer import generatePoseFromState
from bark.runtime.viewer.video_renderer import VideoRenderer

import bark.core
import bark.core.models.behavior
from bark.core.models.dynamic import StateDefinition
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent

from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

def configure_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_dir', "--ed", type=str)
    return parser.parse_args(sys.argv[1:])


def main():
    args = configure_args()
    exp_dir = args.exp_dir or "results/training/toy_evaluation"
    params_filename = glob.glob(os.path.join(exp_dir, "params_*"))
    params = ParameterServer(filename=params_filename[0])
    behavior_params_filename = glob.glob(os.path.join(exp_dir, "behavior_params*"))
    if behavior_params_filename:
      splits = 8
      behavior_params = ParameterServer(filename=behavior_params_filename[0])
      behavior_space = BehaviorSpace(behavior_params)
      hypothesis_set, _ = behavior_space.create_hypothesis_set_fixed_split(split=splits)
      observer = BeliefObserver(params, hypothesis_set, splits=splits)
      behavior = BehaviorDiscreteMacroActionsML(behavior_params)
    else:
      behavior = BehaviorDiscreteMacroActionsML(params)
      observer = NearestAgentsObserver(params)

    evaluator = GoalReached(params)

    scenario_params = ParameterServer(filename="configuration/database/scenario_sets/interaction_merging_light_dense_1D.json")
    scenario_generator = ConfigurableScenarioGeneration(params=scenario_params, num_scenarios=5)
    scenario_file = glob.glob(os.path.join(exp_dir, "scenarios_list*"))
    scenario_generator.load_scenario_list(scenario_file[0])
    viewer = MPViewer(params=params,
                      x_range=[-35, 35],
                      y_range=[-35, 35],
                      follow_agent_id=True)
    env = HyDiscreteHighway(behavior=behavior,
                            observer = observer,
                            evaluator = evaluator,
                            viewer=viewer,
                            scenario_generation=scenario_generator,
                            render=True)

    env.reset()
    actions = [0, 1, 2, 3, 4, 5, 6]
    for action in actions:
      print(action)
      env.step(action)
    agent = FQFAgent(env=env, test_env=env, params=params)

    agent.load_models(os.path.join(exp_dir, "agent/checkpoints/final"))
    # agent.evaluate()


if __name__ == '__main__':
    main()