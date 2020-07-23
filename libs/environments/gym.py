import gym
from gym.envs.registration import register

from bark_ml.environments.single_agent_runtime import SingleAgentRuntime

from bark_ml.environments.blueprints.highway.highway import DiscreteHighwayBlueprint
from bark_ml.environments.blueprints.merging.merging import DiscreteMergingBlueprint
from bark_ml.environments.blueprints.intersection.intersection import DiscreteIntersectionBlueprint


class HyDiscreteHighway(SingleAgentRuntime, gym.Env):

    def __init__(self, params, num_scenarios=20, random_seed=0, viewer=False):
        self._blueprint = DiscreteHighwayBlueprint(params,
                                                   number_of_senarios=num_scenarios,
                                                   random_seed=0)
        SingleAgentRuntime.__init__(self, blueprint=self._blueprint, render=True)


class HyDiscreteMerging(SingleAgentRuntime, gym.Wrapper):

    def __init__(self, params):
        self._blueprint = DiscreteMergingBlueprint(params)
        SingleAgentRuntime.__init__(self, blueprint=self._blueprint, render=True)


class HyDiscreteIntersection(SingleAgentRuntime, gym.Wrapper):

    def __init__(self, params):
        self._blueprint = DiscreteIntersectionBlueprint(params)
        SingleAgentRuntime.__init__(self, blueprint=self._blueprint, render=True)


# register gym envs

register(
    id="hyhighway-v0",
    entry_point="hythe.modules.environments.gym:HyDiscreteHighway"
)

register(
    id="hy-merging-v0",
    entry_point="hythe.modules.environments.gym:HyDiscreteMerging"
)

register(
    id="hy-intersection-v0",
    entry_point="hythe.modules.environments.gym:HyDiscreteIntersection"
)