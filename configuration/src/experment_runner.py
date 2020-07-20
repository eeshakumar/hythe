from argparse import ArgumentParser
import os
import datetime
import yaml
from hythe.modules.experiments.experiment import Experiment
from hythe.modules.environments.gym import HyDiscreteHighway

from fqf_iqn_qrdqn.agent import FQFAgent
from bark_project.modules.runtime.commons.parameters import ParameterServer


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('../fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='hy-highway')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def run(env):
    args = configure_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    agent = FQFAgent(env=env, test_env=env, log_dir=log_dir, seed=)
    agent.run()

def main():
    params = ParameterServer()
    env = HyDiscreteHighway(params=params)
    run(env)
    return

if __name__ == '__main__':
    main()