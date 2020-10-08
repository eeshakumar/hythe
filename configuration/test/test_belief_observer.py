import os
import unittest
from bark.runtime.viewer.matplotlib_viewer import MPViewer

from bark.runtime.commons.parameters import ParameterServer
from hythe.libs.observer.belief_observer import BeliefObserver
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import ConfigurableScenarioGeneration
from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.evaluators.goal_reached import GoalReached
from hythe.libs.environments.gym import HyDiscreteHighway
from hythe.libs.timer.timer import timer
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


class HyBeliefObserverTests(unittest.TestCase):

    # @timer
    def test_belief_observer(self):
        dir_prefix=""
        params = ParameterServer(
            filename="configuration/params/fqf_params.json")
        params["Experiment"]["dir"] = "output/tests"
        params["ML"]["BaseAgent"]["SummaryPath"] = os.path.join(params["Experiment"]["dir"], "agent/summaries")
        params["ML"]["BaseAgent"]["CheckpointPath"] = os.path.join(params["Experiment"]["dir"], "agent/checkpoints")
        # params_scenario_generation = ParameterServer(filename="configuration/database/scenario_sets/interaction_merging_light_dense_behavior_space_default.json")
        splits = 8
        params_behavior = ParameterServer(
            filename="configuration/params/default_params_behavior_space.json")
        behavior_space = BehaviorSpace(params_behavior)
        hypothesis_set, hypothesis_params = behavior_space.create_hypothesis_set_fixed_split(split=splits)
        observer = BeliefObserver(params, hypothesis_set, splits=splits)
        behavior = BehaviorDiscreteMacroActionsML(params_behavior)
        evaluator = GoalReached(params)
        # scenario_generation = ConfigurableScenarioGeneration(num_scenarios=2, params=params_scenario_generation)
        viewer = MPViewer(params=params,
                          x_range=[-35, 35],
                          y_range=[-35, 35],
                          follow_agent_id=True)

        # database creation
        dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=2,
                                 num_serialize_scenarios=2)
        dbs.process(os.path.join(dir_prefix, "configuration/database"),
                    filter_sets="interaction_merging_light_dense_behavior_space_default")
        local_release_filename = dbs.release(version="test", sub_dir="hy_bark_packaged_databases")
        db = BenchmarkDatabase(database_root=local_release_filename)
        scenario_generator, _, _ = db.get_scenario_generator(0)
        env = HyDiscreteHighway(params=params,
                                scenario_generation=scenario_generator,
                                behavior=behavior,
                                evaluator=evaluator,
                                observer=observer,
                                viewer=viewer,
                                render=False)
        max_num_agents = observer._max_num_vehicles
        assert max_num_agents * len(hypothesis_set) == observer.max_beliefs
        agent = FQFAgent(env=env, test_env=env, params=params)
        done = False
        i = 0
        env.reset()
        while i<100:
            action = agent.explore()
            obs, reward, done, _ = agent.env.step(action)
            i += 1

        print("Num actions", i)


# import cProfile


if __name__ == '__main__':
  # cProfile.run('unittest.main()')
    unittest.main()