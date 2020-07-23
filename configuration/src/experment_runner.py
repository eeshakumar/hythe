from argparse import ArgumentParser
import os
from datetime import datetime
import yaml
from hythe.libs.experiments.experiment import Experiment
from hythe.libs.environments.gym import HyDiscreteHighway

from fqf_iqn_qrdqn.agent import FQFAgent
from bark_project.modules.runtime.commons.parameters import ParameterServer

from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('../fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='hyhighway-v0')
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
    agent = FQFAgent(env=env, test_env=env, log_dir=log_dir, seed=0,
                     cuda=args.cuda, **config)
    return agent


def run_single_episode(params, env):
    agent = configure_agent(env)

    exp = Experiment(params=params, agent=agent)
    exp.run_single_episode()


def run(params, env):
    agent = configure_agent(env)

    exp = Experiment(params=params, agent=agent)
    exp.run()


def main():
    params = ParameterServer()
    map_filename = "bark_ml/environments/blueprints/highway/city_highway_straight.xodr"
    num_scenarios = 25
    random_seed = 0
    behavior = BehaviorDiscreteMacroActionsML(params)
    evaluator = GoalReached(params)
    observer = NearestAgentsObserver(params)
    env = HyDiscreteHighway(params=params, num_scenarios=num_scenarios,
                            random_seed=random_seed, behavior=behavior,
                            evaluator=evaluator, observer=observer,
                            map_filename=map_filename)

    run(params, env)
    return


if __name__ == '__main__':
    main()
