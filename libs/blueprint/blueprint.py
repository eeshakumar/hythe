from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark_project.modules.runtime.viewer.matplotlib_viewer import MPViewer

from bark_ml.environments.blueprints.highway.highway import HighwayLaneCorridorConfig
from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_project.modules.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase


class HyHighwayBlueprint(Blueprint):

    def __init__(self,
               params=None,
               map_filename=None,
               num_scenarios=20,
               random_seed=0,
               behavior=None,
               evaluator=None,
               observer=None,
               viewer=False):
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
        self._scenario_generation = ConfigWithEase(
            num_scenarios=num_scenarios,
            map_file_name= map_filename,
            random_seed=random_seed,
            params=params,
            lane_corrido_configs=[left_lane, right_lane]
        )

        if viewer:
            viewer = MPViewer(params=params,
                              x_range=[-35, 35],
                              y_range=[-35, 35],
                              follow_agent_id=True)

        dt = 0.1
        self.evaluator = evaluator
        self._observer = observer
        self._behavior = behavior