import argparse
import os
import yaml
import numpy as np
from datetime import datetime
import gym
from fqf_iqn_qrdqn.agent import FQFAgent
from bark_project.modules.runtime.\
    commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
# from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.environments.gym import DiscreteHighwayGym, ContinuousHighwayGym


# env_id = 'highway-v0'
# env = gym.make(env_id)
#
# env.reset()
#
# test_env = gym.make('highway-v1')

# bark_params = ParameterServer()
# bp = ContinuousHighwayBlueprint(bark_params,
#                               number_of_senarios=10,
#                               random_seed=0,
#                               viewer=False)
# 
# observer = NearestAgentsObserver(bark_params)
env = DiscreteHighwayGym()

test_env = DiscreteHighwayGym()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('./fqf-bark.runfiles/fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='highway-v0')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    agent = FQFAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)

    agent.run()


if __name__ == '__main__':
    get_args()
