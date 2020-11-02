py_binary(
    name = "agent_database_evaluation",
    srcs = [
        "agent_database_evaluation.py",
    ],
    data = [
        "//configuration/database",
    ],
    deps = [
        "@benchmark_database//serialization:database_serializer",
          "@bark_project//bark/benchmark:benchmark_runner_mp",
          "@benchmark_database//load:benchmark_database",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
    ],
)
