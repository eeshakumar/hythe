import sys
import os
from collections import OrderedDict

import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import seaborn as sns

import bark.core
import bark.core.models.behavior
from bark.runtime.commons.parameters import ParameterServer

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer
from hythe.libs.observer.belief_observer import BeliefObserver
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
import yaml
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import ConfigurableScenarioGeneration

from hythe.libs.experiments.experiment import Experiment
from hythe.libs.environments.gym import HyDiscreteHighway
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


is_local = True


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--output_dir', "--od", type=str)
    parser.add_argument('--episode_no', "--en", type=int, default=1000)
    return parser.parse_args(sys.argv[1:])


def configure_behavior_space(params):
    return BehaviorSpace(params)


def main():
    print("Experiment server at :", os.getcwd())

    args = configure_args()
    #load exp params
    exp_dir = args.output_dir
    import glob
    params_filename = glob.glob(os.path.join(exp_dir, "params_[!behavior]*"))
    print(params_filename)
    params = ParameterServer(filename=params_filename[0])
    params.load(fn=params_filename[0])
    params["ML"]["BaseAgent"]["SummaryPath"] = os.path.join(exp_dir, "agent/summaries")
    params["ML"]["BaseAgent"]["CheckpointPath"] = os.path.join(exp_dir, "agent/checkpoints")
    splits = 8
    params_behavior = ParameterServer(filename=os.path.join("configuration/params/1D_desired_gap_no_prior.json"))
    behavior_space = configure_behavior_space(params_behavior)

    hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
    observer = BeliefObserver(params, hypothesis_set, splits=splits)
    behavior = BehaviorDiscreteMacroActionsML(params_behavior)
    evaluator = GoalReached(params)

    viewer = MPViewer(params=params,
                      x_range=[-35, 35],
                      y_range=[-35, 35],
                      follow_agent_id=True)


    # database creation
    dir_prefix = ""
    dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=2,
                             num_serialize_scenarios=10)
    dbs.process(os.path.join(dir_prefix, "configuration/database"), filter_sets="**/**/interaction_merging_light_dense_1D.json")
    local_release_filename = dbs.release(version="test")
    db = BenchmarkDatabase(database_root=local_release_filename)
    scenario_generator, _, _ = db.get_scenario_generator(0)

    video_renderer = VideoRenderer(renderer=viewer, world_step_time=0.2)
    env = HyDiscreteHighway(params=params,
                            scenario_generation=scenario_generator,
                            behavior=behavior,
                            evaluator=evaluator,
                            observer=observer,
                            viewer=video_renderer,
                            render=is_local)
    print(env.action_space)

    # non-agent evaluation mode
    num_steps = 100
    num_samples = params_behavior["BehaviorSpace"]["Hypothesis"]["BehaviorHypothesisIDM"]["NumSamples"]
    print("Steps, samples, splits", num_steps, num_samples, splits)
    step = 1
    env.reset()
    print(env._scenario._eval_agent_ids)

    
    beliefs_df = pd.DataFrame(columns=["Step", "Action", "Agent", "Beliefs", "HyNum"])
    while step <= num_steps:
        action = 1#np.random.randint(0, behavior.action_space.n)
        next_state, reward, done, info = env.step(action)
        for agent, beliefs in observer.beliefs.items():
            for i, belief in enumerate(beliefs):
                beliefs_df = beliefs_df.append({"Step": step, "Action": action, "Agent": agent, "Beliefs": belief, "HyNum": i}, ignore_index=True)
        len_beliefs, beliefs_arr = observer.beliefs_array()
        step += 1

    beliefs_data_filename = "beliefs_{}_{}_{}".format(splits, num_samples, num_steps)
    print(beliefs_df.shape)
    beliefs_df.to_pickle(os.path.join(str(Path.home()), "master_thesis/code/hythe/beliefs_data/", beliefs_data_filename))
    video_filename = os.path.join(str(Path.home()), "master_thesis/code/hythe/beliefs_data/", "video_{}".format(num_samples))
    print(video_filename)
    video_renderer.export_video(filename=video_filename)
    return


if __name__ == '__main__':
    main()