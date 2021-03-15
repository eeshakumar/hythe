py_binary(
    name = "hy-x-dispatch",
    srcs = [
        "hy_x_dispatch.py",
    ],
    data = [
    ],
    imports = [
        "../bark/",
        "../bark_ml/",
        "../python/",
    ],
    main = "hy_x_dispatch.py",
    deps = [
        "@hythe//libs/dispatch:dispatcher",
        "@hythe//libs/dispatch:remote_dispatcher",
    ],
)

py_binary(
    name = "hy-x-run",
    srcs = [
        "x_experiments_runner.py",
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
    main = "x_experiments_runner.py",
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
    name = "hy-exp-load",
    srcs = [
        "experiment_loader.py",
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
    main = "experiment_loader.py",
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