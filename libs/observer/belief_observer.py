# Authors: Eesha Kumar
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
from gym import spaces
from collections import OrderedDict
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
        self.is_enabled_threshold = params["ML"]["BeliefObserver"]["EnableThreshold", "If Threshold to cutoff beliefs must be enabled", True]
        self.threshold = self.params["ML"]["BeliefObserver"]["Threshold", "Cutoff threshold for beliefs", 0.05]
        self.is_discretize = params["ML"]["BeliefObserver"]["Discretize", "If value of belief must be discretized", False]
        self.num_buckets = params["ML"]["BeliefObserver"]["NumBuckets", "Number of discrete values to map beliefs", 8]
        self.is_ciel = params["ML"]["BeliefObserver"]["CielValues", "If to Ceil or Floor value", True]
        self.buckets = np.linspace(0., 1., self.num_buckets)
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
        if self.is_enabled_threshold:
          beliefs_arr = self.threshold_beliefs(beliefs_arr)
        if self.is_discretize:
          beliefs_arr = self.discretize_beliefs(beliefs_arr)
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

    def threshold_beliefs(self, beliefs_arr):
        beliefs_arr[beliefs_arr <= self.threshold] = 0.
        return beliefs_arr

    def discretize_beliefs(self, beliefs_arr):
        bin_idxs = np.digitize(beliefs_arr, self.buckets, right=self.is_ciel)
        assert bin_idxs.shape[0] == beliefs_arr.shape[0]
        return self.buckets[bin_idxs]

    @property
    def max_num_agents(self):
        return self._max_num_vehicles

    @property
    def max_beliefs(self):
        return self.len_beliefs

