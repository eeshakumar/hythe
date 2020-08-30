import os
print(os.environ["PYTHONPATH"])

import bark.core
import bark.core.models.behavior
from bark.runtime.commons.parameters import ParameterServer
from hythe.libs.observer.belief_observer import BeliefObserver
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML
import yaml
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace

from hythe.libs.experiments.experiment import Experiment
from hythe.libs.environments.gym import HyDiscreteHighway
from fqf_iqn_qrdqn.agent import FQFAgent




def configure_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('external/fqn/config', 'fqf.yaml'))
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
    agent = FQFAgent(env=env, test_env=env, log_dir=log_dir, seed=args.seed,
                     cuda=args.cuda, **config)
    return agent


def configure_behavior_space(params):
    return BehaviorSpace(params)


def configure_params():
    params = ParameterServer()
    import uuid
    experiment_seed = str(uuid.uuid4())
    params["Experiment"]["random_seed"] = experiment_seed
    params["Experiment"]["dir"] = str(Path.home().joinpath(".cache/output/experiments/exp_{}".format(experiment_seed)))
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
    params = configure_params()
    behavior_space = configure_behavior_space(params)

    hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=3)
    observer = BeliefObserver(params, hypothesis_set)
    behavior = BehaviorDiscreteML(params)
    evaluator = GoalReached(params)

    num_scenarios = 10
    random_seed = 0
    env = HyDiscreteHighway(params=params, num_scenarios=num_scenarios,
                            random_seed=random_seed, behavior=behavior,
                            evaluator=evaluator, observer=observer,
                            map_filename=params["Experiment"]["map_filename"])

    run(params, env)
    return


if __name__ == '__main__':
    main()