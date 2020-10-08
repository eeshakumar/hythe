# Authors: Eesha Kumar
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
from gym import spaces
from collections import OrderedDict
# from bark_mcts.models.behavior.belief_calculator.belief_calculator import BeliefCalculator
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark.core.world import World, MakeTestWorldHighway, ObservedWorld
from bark.core.models.behavior import *



class BeliefObserver(NearestAgentsObserver):

    def __init__(self, params, hypothesis_set, splits):
        """
        Initialize attributes.
        :param hypothesis_set: hypothesis set created from behaviour space.
        :param splits: number of splits or partitions generated for scenario behaviours.
        """
        super(BeliefObserver, self).__init__(params)
        self.params = params
        self.hypothesis_set = hypothesis_set
        self.splits = splits
        self.belief_calculator = BeliefCalculator(params, hypothesis_set)
        self.beliefs = None
        self.len_beliefs = self._max_num_vehicles * len(hypothesis_set)

    def Reset(self, world):
        """
        Reset world for changing scenarios.
        :param world:
        :return:
        """
        world = super().Reset(world)
        self.belief_calculator = BeliefCalculator(self.params, self.hypothesis_set)
        return world

    def Observe(self, world):
        """
        Transform from world state to network's input state.
        :param world: Observed world.
        :return: Concatenated state with beliefs.
        """
        concatenated_state, agent_ids_by_nearest = super(BeliefObserver, self).Observe(
            world, is_ret_agent_idx_nearest=True)
        self.belief_calculator.BeliefUpdate(world)
        # beliefs is a dictionary mapping every agent id to list of beliefs.
        # the agent ids are ordered by distance.
        beliefs = self.belief_calculator.GetBeliefs()
        self.beliefs = self.order_beliefs(beliefs, agent_ids_by_nearest)
        len_beliefs, beliefs_arr = self.beliefs_array()
        assert len_beliefs == self.len_beliefs
        beliefs_concatenated_state = np.concatenate([concatenated_state, beliefs_arr])
        return beliefs_concatenated_state

    def beliefs_array(self):
        """
        Helper to flatten dictionary to lists.
        :return:
        """
        beliefs_list = []
        for k, v in self.beliefs.items():
            beliefs_list.extend(v)
        return len(beliefs_list), np.asarray(beliefs_list)

    def order_beliefs(self, beliefs, agent_ids_by_nearest):
        """
        Order beliefs based on nearest agent. I.e. the first agent id (key) of ordered_beliefs is the nearest agent.
        :param beliefs: beliefs as calculated by belief calculator.
        :param agent_ids_by_nearest: list of agent_ids sorted by distance
        :return:
        """
        ordered_beliefs = OrderedDict({agent_id: beliefs[agent_id] for agent_id in agent_ids_by_nearest})
        return ordered_beliefs

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.zeros(self.len_beliefs + self._len_ego_state + \
        self._max_num_vehicles*self._len_relative_agent_state),
            high=np.zeros(self.len_beliefs + self._len_ego_state + \
        self._max_num_vehicles*self._len_relative_agent_state)
        )

    @property
    def max_num_agents(self):
        return self._max_num_vehicles

    @property
    def max_beliefs(self):
        return self.len_beliefs

