py_binary(
    name = "agent_database_evaluation",
    srcs = [
        "agent_database_evaluation.py",
    ],
    data = [
        "//configuration/database",
    ],
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
