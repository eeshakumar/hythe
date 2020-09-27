# Authors: Eesha Kumar
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
from collections import OrderedDict
# from bark_mcts.models.behavior.belief_calculator.belief_calculator import BeliefCalculator
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark.core.models.behavior import *


class BeliefObserver(NearestAgentsObserver):

    def __init__(self, params, hypothesis_set, splits):
        """
        Initialize attributes.
        :param hypothesis_set: hypothesis set created from behaviour space.
        :param splits: number of splits or partitions generated for scenario behaviours.
        """
        super(BeliefObserver, self).__init__(params)
        self.hypothesis_set = hypothesis_set
        self.splits = splits
        self.belief_calculator = BeliefCalculator(params, hypothesis_set)
        self.beliefs = self.belief_calculator.GetBeliefs()

    def Reset(self, world):
        """
        Reset world for changing scenarios.
        :param world:
        :return:
        """
        super().Reset(world)
        self.belief_calculator = BeliefCalculator(hypothesis_set)
        return

    def Observe(self, world):
        """
        Transform from world state to networs input state.
        :param world: Observed world.
        :return: Concatenated state with beliefs.
        """
        concatenated_state, agent_ids_by_nearest = super(BeliefObserver, self).Observe(world)
        self.belief_calculator = self.belief_calculator.BeliefUpdate(world)
        # beliefs is a dictionary mapping every agent id to list of beliefs.
        # the agent ids are ordered by distance.
        beliefs = self.belief_calculator.GetBeliefs()
        self.beliefs = BeliefObserver.order_beliefs(agent_ids_by_nearest)
        len_beliefs, beliefs_arr = self.beliefs_array()
        beliefs_concatenated_state = np.zeros(concatenated_state.shape[0] + len_beliefs)
        beliefs_concatenated_state = np.concatenate([concatenated_state, beliefs_arr])
        return beliefs_concatenated_state

    def beliefs_array(self):
        """
        Helper to flatten dictionary to lists.
        :return:
        """
        len_beliefs = 0
        beliefs_list = []
        for k, v in self.beliefs.items():
            len_beliefs += len(v)
            beliefs_list.extend(v)
        assert len_beliefs == len(beliefs_list)
        return len_beliefs, np.asarray(beliefs_list)

    def order_beliefs(self, agent_ids_by_nearest):
        """
        Order beliefs based on nearest agent. I.e. the first agent id (key) of ordered_beliefs is the nearest agent.
        :param beliefs:
        :param agent_ids_by_nearest:
        :return:
        """
        ordered_beliefs = OrderedDict({agent_id:self.beliefs[agent_id] for agent_id in agent_ids_by_nearest})
        return ordered_beliefs

