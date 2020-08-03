py_library(
    name = "test_bark_imports",
    srcs = [
        "test_bark_imports/some_bark_imports.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "test_imports",
    srcs = [
        "test_imports/some_imports.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "libs",
    srcs = glob([
        "experiments/*/*.py",
        "environments/*/*.py",
        "dispatch/*/*.py",
    ]),
    deps = "//*/*.py",
)
