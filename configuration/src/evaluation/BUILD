py_binary(
    name = "agent_database_evaluation",
    srcs = [
        "agent_database_evaluation.py",
    ],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "//results/training:training"
    ],
    imports = [
        "../bark/",
        "../bark_mcts/",
        "../bark_mcts/python_wrapper/",
        "../bark_ml/",
        "../bark_ml/python_wrapper",
        "../python/",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "@bark_project//bark/benchmark:benchmark_runner_mp",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@hythe//libs/blueprint",
        "@hythe//libs/environments:gym",
        "@hythe//libs/observer:belief_observer",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "//configuration/src/common:common"
    ],
)

py_binary(
    name = "agent_evaluation",
    srcs = [
        "agent_evaluation.py",
    ],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "//results/training:training"
    ],
    imports = [
        "../bark/",
        "../bark_mcts/",
        "../bark_mcts/python_wrapper/",
        "../bark_ml/",
        "../bark_ml/python_wrapper",
        "../python/",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "@bark_project//bark/benchmark:benchmark_runner_mp",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@hythe//libs/blueprint",
        "@hythe//libs/environments:gym",
        "@hythe//libs/evaluation:evaluation",
        "@hythe//libs/observer:belief_observer",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "//configuration/src/common:common"
    ],
)

py_binary(
    name = "evaluate_experiment",
    srcs = [
        "evaluate_experiment.py",
    ],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "@bark_ml_project//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
        "//results/training:training",
    ],
    imports = [
        "../bark/",
        "../bark_mcts/",
        "../bark_mcts/python_wrapper/",
        "../bark_ml/",
        "../bark_ml/python_wrapper",
        "../python/",
    ],
    main = "evaluate_experiment.py",
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
        "//configuration/src/common:common",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@hythe//libs/observer:belief_observer",
    ],
)
