py_library(
    name = "experiment",
    srcs = [
        "experiment.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "experiment_manager",
    srcs = [
        "experiment_manager.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@hythe//libs/dispatch:dispatcher",
        "@hythe//libs/dispatch:sequential_dispatcher",
    ],
)
