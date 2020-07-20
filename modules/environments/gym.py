import gym
from gym.envs.registration import register

from bark_ml.environments.single_agent_runtime import SingleAgentRuntime

from bark_ml.environments.blueprints.highway.highway import DiscreteHighwayBlueprint
from bark_ml.environments.blueprints.merging.merging import DiscreteMergingBlueprint
from bark_ml.environments.blueprints.intersection.intersection import DiscreteIntersectionBlueprint


class HyDiscreteHighway(SingleAgentRuntime, gym.Wrapper):

    def __init__(self, params):
        self._blueprint = DiscreteHighwayBlueprint(params)
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
    id="hy-highway",
    entry_point="hythe.modules.environments.gym:HyDiscreteHighway"
)

register(
    id="hy-merging",
    entry_point="hythe.modules.environments.gym:HyDiscreteMerging"
)

register(
    id="hy-intersection",
    entry_point="hythe.modules.environments.gym:HyDiscreteIntersection"
)