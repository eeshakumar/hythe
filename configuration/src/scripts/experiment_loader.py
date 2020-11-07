from argparse import ArgumentParser
import yaml
import os
from pathlib import Path
from datetime import datetime
from hythe.libs.experiments.experiment import Experiment
from fqf_iqn_qrdqn.agent import FQFAgent

episode_num = 2
exp_dirname = "~/.cache/output/experiments/exp.175/"


def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('../fqn/config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='hyhighway-v0')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=122)
    return parser.parse_args()


def configure_agent(env, is_loaded=False):
    args = configure_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    agent = FQFAgent(env=env, test_env=env, log_dir=log_dir, seed=args.seed,
                     cuda=args.cuda, **config)
    if is_loaded:
        agent.load_models(exp_dirname)
    return agent


def main():
    assert Experiment.is_experiment(exp_dir=exp_dirname)
    params_files, scenario_files = Experiment.load(exp_dirname)
    for file in params_files:
        if int(file.split(".")[-2].split("_")[-1]) == episode_num:
            episode_params = file
            break

    for file in scenario_files:
        if int(file.split("_")[-1]) == episode_num:
            episode_scenario = file
            break

    params, blueprint = Experiment.restore_blueprint(episode_params, episode_scenario)
    env = Experiment.restore_env(params=params, blueprint=blueprint)
    agent = configure_agent(env, True)

    exp = Experiment(params, agent)
    exp.run_single_episode(episode_num=episode_num, is_loaded=True)


if __name__ == '__main__':
    main()
