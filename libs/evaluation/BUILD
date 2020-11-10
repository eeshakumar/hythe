py_library(
    name = "evaluation",
    srcs = [
        "training_benchmark_database.py"
    ],
    data = [
        "@bark_project//bark:generate_core",
    ],
    deps = [
        "@bark_project//bark/benchmark:benchmark_runner",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
    ],
    visibility = ["//visibility:public"],
)
