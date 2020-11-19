try:
    import debug_settings
except:
    pass
import unittest

import bark.core.commons
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from hythe.libs.observer.belief_observer import BeliefObserver

import bark.core
import bark.core.models.behavior


from bark.runtime.commons.parameters import ParameterServer

def pickle_unpickle(obj):
  import pickle
  pkl = "{obj.__name__}.pickle"
  with open(pkl, "wb") as f:
    pickle.dump(obj, f)
  obj2 = None
  with open(pkl, "rb") as f:
    obj2 = pickle.load(f)
  return obj2


class TestBeliefObserverPickle(unittest.TestCase):

  def test_pickle_belief_observer(self):
    params = ParameterServer()
    # init behavior space
    splits = 2
    behavior_params = ParameterServer()
    behavior_space = BehaviorSpace(behavior_params)
    hypothesis_set, _ = behavior_space.create_hypothesis_set_fixed_split(split=splits)
    observer = BeliefObserver(params, hypothesis_set, splits=splits)

    po = pickle_unpickle(observer)
    self.assertIsNotNone(po)
    self.assertTrue(isinstance(po, BeliefObserver))
    self.assertEqual(po.splits, observer.splits)
    self.assertEqual(po.len_beliefs, observer.len_beliefs)


if __name__ == '__main__':
    unittest.main()