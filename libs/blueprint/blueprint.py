from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_project.bark.runtime.viewer.matplotlib_viewer import MPViewer

from bark_ml.environments.blueprints.highway.highway import HighwayLaneCorridorConfig
from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_project.bark.runtime.scenario.scenario_generation.config_with_ease import \
    LaneCorridorConfig, ConfigWithEase

from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML


class HyHighwayBlueprint(Blueprint):

    def __init__(self,
                 params=None,
                 map_filename=None,
                 num_scenarios=20,
                 random_seed=0,
                 behavior=None,
                 evaluator=None,
                 observer=None,
                 scenario_generation=None,
                 viewer=True):
        if scenario_generation is None:
            left_lane = HighwayLaneCorridorConfig(params=params,
                                                  road_ids=[16],
                                                  lane_corridor_id=0,
                                                  min_vel=25.0,
                                                  max_vel=30.0,
                                                  controlled_ids=None)
            right_lane = HighwayLaneCorridorConfig(params=params,
                                                   road_ids=[16],
                                                   lane_corridor_id=1,
                                                   min_vel=25.0,
                                                   max_vel=30.0,
                                                   controlled_ids=True)
            scenario_generation = ConfigWithEase(
                num_scenarios=num_scenarios,
                map_file_name=map_filename,
                random_seed=random_seed,
                params=params,
                lane_corridor_configs=[left_lane, right_lane]
            )

        if viewer:
            viewer = MPViewer(params=params,
                              x_range=[-35, 35],
                              y_range=[-35, 35],
                              follow_agent_id=True)

        dt = 0.1

        Blueprint.__init__(self,
                           scenario_generation=scenario_generation,
                           viewer=viewer,
                           dt=dt,
                           evaluator=evaluator,
                           observer=observer,
                           ml_behavior=behavior)


class HyHighwayDiscreteBlueprint(HyHighwayBlueprint):

    def __init__(self,
                 params=None,
                 map_filename=None,
                 num_scenarios=20,
                 random_seed=0,
                 behavior=None,
                 evaluator=None,
                 observer=None,
                 scenario_generation=None,
                 viewer=True):
        if behavior is None:
            behavior = BehaviorDiscreteMacroActionsML(params)

        if evaluator is None:
            evaluator = GoalReached(params)

        if observer is None:
            observer = NearestAgentsObserver(params)

        HyHighwayBlueprint.__init__(self,
                                    params=params,
                                    map_filename=map_filename,
                                    num_scenarios=num_scenarios,
                                    random_seed=random_seed,
                                    behavior=behavior,
                                    evaluator=evaluator,
                                    observer=observer,
                                    scenario_generation=scenario_generation,
                                    viewer=viewer)

    @property
    def scenario_generation(self):
        return self._scenario_generation

    @property
    def behavior(self):
        return self._behavior

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def observer(self):
        return self._observer
