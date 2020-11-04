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
