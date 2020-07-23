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
    name = "hy-exp-run",
    srcs = [
        "src/experment_runner.py",
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
    main = "src/experment_runner.py",
    deps = [
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
