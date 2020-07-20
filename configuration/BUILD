py_binary(
    name = "configuration-imports-bin",
    main = "src/main.py",
    srcs = [
         "src/main.py",
    ],
    imports = ['../python/'],
    deps = [
      "@bark_project//modules/runtime:runtime",
      "@bark_ml//bark_ml/behaviors:behaviors"
    ],
)

py_binary(
    name = "train-fqf",
    main = "src/fqf.py",
    srcs = [
         "src/fqf.py",
    ],
    imports = ['../python/'],
    data = ["@fqn//:fqn_config"],
    deps = [
      "@fqn//:pylib_fqn"
    ],
)

py_binary(
    name = "hy-exp-run",
    main = "src/experment_runner.py",
    srcs = [
         "src/experment_runner.py",
    ],
    imports = ['../bark/', '../bark_ml/', '../python/'],
    data = ["@fqn//:fqn_config",
            "@bark_project//python:bark.so",
            "@bark_ml_project//python:bark_ml_library.so"],
    deps = [
      "@fqn//:pylib_fqn",
      "@bark_ml_project//bark_ml/environments:gym",
      "@bark_ml_project//bark_ml/environments:blueprints",
      "@hythe//modules/environments:gym",
      "@hythe//modules/experiments:experiment",
    ],
)

py_binary(
    name = "fqf-bark",
    main = "src/fqf_bark.py",
    srcs = [
         "src/fqf_bark.py",
    ],
    imports = ['../bark/', '../bark_ml/', '../python/'],
    data = ["@fqn//:fqn_config",
            "@bark_project//python:bark.so",
            "@bark_ml_project//python:bark_ml_library.so"],
    deps = [
      "@fqn//:pylib_fqn",
      "@bark_ml_project//bark_ml/environments:gym",
      "@bark_ml_project//bark_ml/environments:blueprints"
    ],
)

py_binary(
    name = "qrdqn-bark",
    main = "src/qrdqn_bark.py",
    srcs = [
         "src/qrdqn_bark.py",
    ],
    imports = ['../bark/', '../bark_ml/', '../python/'],
    data = ["@fqn//:fqn_config",
            "@bark_project//python:bark.so",
            "@bark_ml_project//python:bark_ml_library.so"],
    deps = [
      "@fqn//:pylib_fqn",
      "@bark_ml_project//bark_ml/environments:gym",
      "@bark_ml_project//bark_ml/environments:blueprints"
    ],
)

py_test(
  name = "continuous_env",
  srcs = ["src/continuous_env.py"],
  data = ["@bark_project//python:bark.so",
          "@bark_ml_project//python:bark_ml_library.so"],
  imports = ['../bark_ml/', '../python/'],
  deps = [
    "@bark_ml_project//bark_ml/environments:gym",
    "@bark_ml_project//bark_ml/environments:blueprints",
  ]
)