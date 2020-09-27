import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from bark.runtime.commons.parameters import ParameterServer

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from hythe.libs.observer.belief_observer import BeliefObserver
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
import yaml
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace

from hythe.libs.experiments.experiment import Experiment
from hythe.libs.environments.gym import HyDiscreteHighway
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


is_local=True


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('external/fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='hy-highway-v0')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=122)
    parser.add_argument('--mode', type=str, default="train")
    return parser.parse_args()


def configure_agent(params, env):
    args = configure_args()

    # with open(args.config) as f:
    #     config = yaml.load(f, Loader=yaml.SafeLoader)

    # name = args.config.split('/')[-1].rstrip('.yaml')
    # time = datetime.now().strftime("%Y%m%d-%H%M")
    # log_dir = os.path.join(
    #     'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    # print('Loggind at', log_dir)
    agent = FQFAgent(env=env, test_env=env, params=params)
    return agent, args


def configure_behavior_space(params):
    return BehaviorSpace(params)


def configure_params(params):
    import uuid
    experiment_seed = str(uuid.uuid4())
    params["Experiment"]["random_seed"] = experiment_seed
    params["Experiment"]["dir"] = str(Path.home().joinpath("output/experiments/exp_{}".format(experiment_seed)))
    Path(params["Experiment"]["dir"]).mkdir(parents=True, exist_ok=True)
    params["Experiment"]["params"] = "params_{}_{}.json"
    params["Experiment"]["scenarios_generated"] = "scenarios_list_{}_{}"
    params["Experiment"]["num_episodes"] = 10
    params["Experiment"]["map_filename"] = "external/bark_ml_project/bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
    return params


def configure_scenario_generation():
    return


def run(params, env):
    agent = configure_agent(env)
    exp = Experiment(params=params, agent=agent)
    exp.run()


def main():
    if is_local:
        dir_prefix = ""
    else:
        dir_prefix = "hy-fqf-exp.runfiles/hythe/"
    print("Experiment server at :", os.getcwd())
    params = ParameterServer(filename=os.path.join(dir_prefix, "configuration/params/fqf_params.json"))
    params = configure_params(params)
    experiment_id = params["Experiment"]["random_seed"]
    params_filename = os.path.join(params["Experiment"]["dir"], "params_{}.json".format(experiment_id))

    # configure belief observer
    splits = 2
    behavior_space = configure_behavior_space(params)
    hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
    observer = BeliefObserver(params, hypothesis_set, splits=splits)

    behavior = BehaviorDiscreteMacroActionsML(params)
    evaluator = GoalReached(params)

    viewer = MPViewer(params=params,
                      x_range=[-35, 35],
                      y_range=[-35, 35],
                      follow_agent_id=True)

    params.Save(filename=params_filename)
    logging.info('-' * 60)
    logging.info("Writing params to :{}".format(params_filename))
    logging.info('-' * 60)

    # database creation
    dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=2,
                             num_serialize_scenarios=100)
    dbs.process(os.path.join(dir_prefix, "configuration/database"), filter_sets="interaction_merging_light_dense")
    local_release_filename = dbs.release(version="test", sub_dir="hy_bark_packaged_databases")
    db = BenchmarkDatabase(database_root=local_release_filename)
    scenario_generator, _, _ = db.get_scenario_generator(0)

    env = HyDiscreteHighway(params=params,
                            scenario_generation=scenario_generator,
                            behavior=behavior,
                            evaluator=evaluator,
                            observer=observer,
                            viewer=viewer,
                            render=False)

    run(params, env)


if __name__ == '__main__':
    main()