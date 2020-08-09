import gym
from gym.envs.registration import register

from bark_ml.environments.single_agent_runtime import SingleAgentRuntime

from bark_ml.environments.blueprints.highway.highway import DiscreteHighwayBlueprint
from bark_ml.environments.blueprints.merging.merging import DiscreteMergingBlueprint
from bark_ml.environments.blueprints.intersection.intersection import DiscreteIntersectionBlueprint

from hythe.libs.blueprint.blueprint import HyHighwayDiscreteBlueprint


class HyDiscreteHighway(SingleAgentRuntime, gym.Env):

    def __init__(self, params=None, num_scenarios=25, random_seed=0, viewer=True,
                 behavior=None, evaluator=None, observer=None, scenario_generation=None,
                 map_filename=None, blueprint=None):
        if blueprint is None:
            self._blueprint = HyHighwayDiscreteBlueprint(params=params,
                                                         map_filename=map_filename,
                                                         num_scenarios=num_scenarios,
                                                         random_seed=random_seed,
                                                         behavior=behavior,
                                                         evaluator=evaluator,
                                                         observer=observer,
                                                         scenario_generation=scenario_generation,
                                                         viewer=viewer)
        else:
            self._blueprint = blueprint
        SingleAgentRuntime.__init__(self, blueprint=self._blueprint, render=True)

    @property
    def blueprint(self):
        return self._blueprint


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
