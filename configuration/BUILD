py_binary(
    name = "cib",
    srcs = [
        "src/main.py",
    ],
    imports = ["../python/"],
    main = "src/main.py",
    deps = [
        "@bark_ml_project//bark_ml/behaviors",
        "@bark_project//modules/runtime",
    ],
)

py_binary(
    name = "train-fqf",
    srcs = [
        "src/fqf.py",
    ],
    data = ["@fqn//:fqn_config"],
    imports = ["../python/"],
    main = "src/fqf.py",
    deps = [
        "@fqn//:pylib_fqn",
    ],
)

py_binary(
    name = "hy-x-dispatch",
    srcs = [
        "src/hy_x_dispatch.py",
    ],
    data = [
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    main = "src/hy_x_dispatch.py",
    deps = [
        "@hythe//libs/dispatch:dispatcher",
        "@hythe//libs/dispatch:remote_dispatcher",
    ],
)

py_binary(
    name = "hy-x-run",
    srcs = [
        "src/x_experiments_runner.py",
    ],
    data = [
        "//configuration/params:hy_params",
        "@bark_ml_project//python:bark_ml_library.so",
        "@bark_project//python:bark.so",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    main = "src/x_experiments_runner.py",
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_project//modules/runtime/viewer:py_viewer",
        "@fqn//:pylib_fqn",
        "@hythe//libs/blueprint",
        "@hythe//libs/dispatch:dispatcher",
        "@hythe//libs/dispatch:parallel_dispatcher",
        "@hythe//libs/environments:gym",
        "@hythe//libs/experiments:experiment",
        "@hythe//libs/experiments:experiment_manager",
    ],
)

py_binary(
    name = "hy-exp-run",
    srcs = [
        "src/experiment_runner.py",
    ],
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
    main = "src/experiment_runner.py",
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

py_binary(
    name = "hy-fqf-exp",
    srcs = [
        "src/hy_fqf_benchmark_runner.py",
    ],
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
        "../bark_ml/python_wrapper",
        "../python/",
    ],
    main = "src/hy_fqf_benchmark_runner.py",
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
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@planner_uct//bark_mcts/runtime/scenario:scenario_set_generation",
    ],
)

py_binary(
    name = "hy-exp-eval",
    srcs = [
        "src/evaluate_experiment.py",
    ],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "@bark_ml_project//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
        "@bark_project//bark/runtime/tests:xodr_data",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    main = "src/evaluate_experiment.py",
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
    ],
)

py_binary(
    name = "hy-x-beliefs",
    srcs = [
        "src/hy_x_beliefs.py",
    ],
    data = [
        "//configuration/database",
        "//configuration/params:hy_params",
        "@bark_project//bark:generate_core",
        "@bark_ml_project//bark_ml:generate_core",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_mcts/",
        "../bark_mcts/python_wrapper/",
        "../bark_ml/",
        "../bark_ml/python_wrapper",
        "../python/",
    ],
    main = "src/hy_x_beliefs.py",
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
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@planner_uct//bark_mcts/runtime/scenario:scenario_set_generation",
        "@hythe//libs/observer:belief_observer",
        "//configuration/src/common:common",
    ],
)

py_binary(
    name = "hy-exp-load",
    srcs = [
        "src/experiment_loader.py",
    ],
    data = [
        "//configuration/params:hy_params",
        "@bark_ml_project//python:bark_ml_library.so",
        "@bark_project//python:bark.so",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    main = "src/experiment_loader.py",
    deps = [
        "@bark_ml_project//bark_ml/behaviors",
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_project//modules/runtime/viewer:py_viewer",
        "@fqn//:pylib_fqn",
        "@hythe//libs/blueprint",
        "@hythe//libs/environments:gym",
        "@hythe//libs/experiments:experiment",
    ],
)

py_binary(
    name = "fqf-bark",
    srcs = [
        "src/fqf_bark.py",
    ],
    data = [
        "@bark_ml_project//python:bark_ml_library.so",
        "@bark_project//python:bark.so",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    main = "src/fqf_bark.py",
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@fqn//:pylib_fqn",
    ],
)

py_binary(
    name = "qrdqn-bark",
    srcs = [
        "src/qrdqn_bark.py",
    ],
    data = [
        "@bark_ml_project//python:bark_ml_library.so",
        "@bark_project//python:bark.so",
        "@fqn//:fqn_config",
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    main = "src/qrdqn_bark.py",
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@fqn//:pylib_fqn",
    ],
)

py_test(
    name = "continuous_env",
    srcs = ["src/continuous_env.py"],
    data = [
        "@bark_ml_project//python:bark_ml_library.so",
        "@bark_project//python:bark.so",
    ],
    imports = [
        "../bark_ml/",
        "../python/",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
    ],
)

py_test(
    name = "test_env",
    srcs = ["test/test_env.py"],
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
    name = "hy-beliefs-tests",
    srcs = ["test/test_belief_observer.py"],
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
    main = "test/test_belief_observer.py",
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
    name = "hy-obs-space-tests",
    srcs = ["test/test_belief_observation_space.py"],
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
    main = "test/test_belief_observation_space.py",
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
