from argparse import ArgumentParser
import os
from pathlib import Path
from datetime import datetime
import yaml
import time
from hythe.libs.experiments.experiment import Experiment
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from hythe.libs.environments.gym import HyDiscreteHighway, GymSingleAgentRuntime

from fqf_iqn_qrdqn.agent import FQFAgent
from bark_project.bark.runtime.commons.parameters import ParameterServer

from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.evaluators.goal_reached import GoalReached

from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('hy-exp-run.runfiles/fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='hy-highway-v0')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=122)
    return parser.parse_args()


def configure_agent(env):
    args = configure_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    print('Loggind at', log_dir)
    agent = FQFAgent(env=env, test_env=env, log_dir=log_dir, seed=args.seed,
                     cuda=args.cuda, **config)
    return agent


def configure_params(params):
    import uuid
    experiment_seed = str(uuid.uuid4())
    params["Experiment"]["random_seed"] = experiment_seed
    params["Experiment"]["dir"] = str(Path.home().joinpath("output/experiments/exp_{}".format(experiment_seed)))
    Path(params["Experiment"]["dir"]).mkdir(parents=True, exist_ok=True)
    params["Experiment"]["params"] = "params_{}_{}.json"
    params["Experiment"]["scenarios_generated"] = "scenarios_list_{}_{}"
    params["Experiment"]["num_episodes"] = 100000
    params["Experiment"]["map_filename"] = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
    return params


def run_single_episode(params, env):
    agent = configure_agent(env)
    exp = Experiment(params=params, agent=agent)
    exp.run_single_episode()


def run(params, env):
    agent = configure_agent(env)
    exp = Experiment(params=params, agent=agent)
    exp.run()


def main():
    print("Experiment server at:", os.getcwd())
    params = ParameterServer(filename="hy-exp-run.runfiles/hythe/configuration/params/default_exp_runne_params.json")
    params = configure_params(params)
    num_scenarios = 5
    random_seed = 0
    behavior = BehaviorDiscreteMacroActionsML(params)
    # evaluator = GoalReachedGuiding(params)
    evaluator = GoalReached(params)
    observer = NearestAgentsObserver(params)
    viewer = MPViewer(params=params,
                        x_range=[-35, 35],
                        y_range=[-35, 35],
                        follow_agent_id=True)
    experiment_id = params["Experiment"]["random_seed"]
    params_filename = "./{}_default_exp_runner_params.json".format(experiment_id)
    print(params_filename)
    params.Save(filename=params_filename)
    # database creation 
    dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=2, num_serialize_scenarios=20) # increase the number of serialize scenarios to 100
    dbs.process("hy-exp-run.runfiles/hythe/configuration/database")
    local_release_filename = dbs.release(version="test", sub_dir="hy_bark_packaged_databases")
    db = BenchmarkDatabase(database_root=local_release_filename)

    # switch this to other generator to get other index
    # scenario_generator, _, _ = db.get_scenario_generator(0)
    scenario_generator, _, _ = db.get_scenario_generator(1)
    #
    # env = GymSingleAgentRuntime(ml_behavior = behavior,
    #                             observer = observer,
    #                             evaluator = evaluator,
    #                             step_time=0.2,
    #                             viewer=viewer,
    #                             scenario_generator=scenario_generator,
    #                             render=False)

    env = HyDiscreteHighway(params=params,
                            scenario_generation=scenario_generator,
                            behavior=behavior,
                            evaluator=evaluator,
                            observer=observer,
                            viewer=viewer,
                            render=False)

    run(params, env)
    # video_renderer.drawWorld(env._world)
    env._viewer.export_video("./test_video")
    # from gym.envs.registration import register
    # register(
    #     id='highway-v1',
    #     entry_point='bark_ml.environments.gym:DiscreteHighwayGym'
    # )
    # import gym
    # env = gym.make("highway-v1")
    # env.reset()
    # actions = [5]*100
    # print(actions)
    # for action in actions:
    #     env.step(action)
    #     time.sleep(0.2)
    return


if __name__ == '__main__':
    main()
