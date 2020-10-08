from argparse import ArgumentParser
import os
from pathlib import Path
from datetime import datetime
import yaml
import time

from bark.runtime.scenario.scenario_generation import ConfigurableScenarioGeneration
from hythe.libs.experiments.experiment import Experiment
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent
from bark_project.bark.runtime.commons.parameters import ParameterServer

from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.evaluators.goal_reached import GoalReached

from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('external/fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='hy-highway-v0')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=122)
    return parser.parse_args()


def configure_agent(env, params):
    args = configure_args()

    # with open(args.config) as f:
    #     config = yaml.load(f, Loader=yaml.SafeLoader)
    #
    # name = args.config.split('/')[-1].rstrip('.yaml')
    # time = datetime.now().strftime("%Y%m%d-%H%M")
    # log_dir = os.path.join(
    #     'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    # print('Loggind at', log_dir)
    agent = FQFAgent(env=env, test_env=env, params=params)
    return agent


def main():
    output_dir = "/home/ekumar/output/experiments"
    exp_id = "exp_f9d3badb-628f-41d2-9a9f-d2d25faf0805"
    exp_dir = os.path.join(output_dir, exp_id)
    import glob
    params_filename = glob.glob(os.path.join(exp_dir, "params_*"))
    print(params_filename)
    params = ParameterServer(filename=params_filename[0])
    # params["ML"]["BaseAgent"]["SummaryPath"] = os.path.join(exp_dir, "agent/summaries")
    # params["ML"]["BaseAgent"]["CheckpointPath"] = os.path.join(exp_dir, "agent/checkpoints")
    # params.load(fn=params_filename)
    behavior = BehaviorDiscreteMacroActionsML(params)
    evaluator = GoalReached(params)
    observer = NearestAgentsObserver(params)
    scenario_generator = ConfigurableScenarioGeneration(params=params, num_scenarios=5)
    scenario_file = glob.glob(os.path.join(exp_dir, "scenarios_list*"))
    print(scenario_file)
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

    agent = configure_agent(env, params)
    # # print(agent.online_net.state_dict())
    # # print(agent.online_net.dqn_net.state_dict())
    agent.load_models(exp_dir)
    # agent.load_models(os.path.join(exp_dir, "agent/checkpoints/final"))
    agent.evaluate()


if __name__ == '__main__':
    main()