
py_test(
    name = "test_env",
    srcs = ["test_env.py"],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "@bark_ml_project//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_project//bark/runtime/viewer:py_viewer",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@fqn//:pylib_fqn",
        "@hythe//libs/blueprint",
        "@hythe//libs/environments:gym",
        "@hythe//libs/experiments:experiment",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@planner_uct//bark_mcts/runtime/scenario:scenario_set_generation",
    ],
)

py_test(
    name = "test_training_benchmark",
    srcs = ["test_training_benchmark.py"],
    data = [
        "//configuration/database",
        "@bark_ml_project//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "@bark_project//bark/runtime/viewer:py_viewer",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@hythe//libs/blueprint",
        "@hythe//libs/environments:gym",
        "@hythe//libs/experiments:experiment",
        "@hythe//libs/evaluation:evaluation",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@planner_uct//bark_mcts/runtime/scenario:scenario_set_generation",
    ],
)


py_test(
    name = "test_belief_observer",
    srcs = ["test_belief_observer.py"],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "@bark_ml_project//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "@bark_project//bark/runtime/viewer:py_viewer",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@fqn//:pylib_fqn",
        "@hythe//libs/blueprint",
        "@hythe//libs/environments:gym",
        "@hythe//libs/experiments:experiment",
        "@hythe//libs/observer:belief_observer",
        "@hythe//libs/timer",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@planner_uct//bark_mcts/runtime/scenario:scenario_set_generation",
    ],
)

py_test(
    name = "test_belief_observation_space",
    srcs = ["test_belief_observation_space.py"],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "@bark_ml_project//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "@bark_project//bark/runtime/viewer:py_viewer",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@fqn//:pylib_fqn",
        "@hythe//libs/blueprint",
        "@hythe//libs/environments:gym",
        "@hythe//libs/experiments:experiment",
        "@hythe//libs/observer:belief_observer",
        "@hythe//libs/timer",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@planner_uct//bark_mcts/runtime/scenario:scenario_set_generation",
    ],
)
