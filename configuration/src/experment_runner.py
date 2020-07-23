from argparse import ArgumentParser
import os
from datetime import datetime
import yaml
from hythe.libs.experiments.experiment import Experiment
from hythe.libs.environments.gym import HyDiscreteHighway

from fqf_iqn_qrdqn.agent import FQFAgent
from bark_project.modules.runtime.commons.parameters import ParameterServer

from bark_ml.environments.blueprints import DiscreteHighwayBlueprint


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('../fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='hyhighway-v0')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=122)
    return parser.parse_args()


def run(env):
    args = configure_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    agent = FQFAgent(env=env, test_env=env, log_dir=log_dir, seed=0,
                     cuda=args.cuda, **config)
    # agent.run()

    exp = Experiment(agent=agent)
    exp.run()


def main():
    # experiment_params = Params(["xonfiguration/params/common_parameters.yaml"])
    params = ParameterServer()
    env = HyDiscreteHighway(params=params, num_scenarios=10,
                            random_seed=0, viewer=False)
    run(env)
    return


if __name__ == '__main__':
    main()
