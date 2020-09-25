load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def hythe_deps():
    _maybe(
        native.new_local_repository,
        name = "python_linux",
        path = "./python/venv/",
        build_file_content = """
cc_library(
    name = "python-lib",
    srcs = glob(["lib/libpython3.*", "libs/python3.lib", "libs/python36.lib"]),
    hdrs = glob(["include/**/*.h", "include/*.h"]),
    includes = ["include/python3.6m", "include", "include/python3.7m", "include/python3.5m"], 
    visibility = ["//visibility:public"],
)
        """,
    )

    _maybe(
        #        native.local_repository,
        native.local_repository,
        name = "bark_ml_project",
        #        commit = "5d81994712ccb6eb73da7420bf487110ed16ebeb",
        #        remote = "https://github.com/bark-simulator/bark-ml",
        path = "/home/ekumar/master_thesis/code/bark-ml",
    )

    _maybe(
        #        git_repository,
        native.local_repository,
        name = "bark_project",
        path = "/home/ekumar/master_thesis/code/bark-drl",
        #        branch = "jb_bark_adaptations",
        #        commit = "5e2ce8893e21f7236ee7c43351e61c0bce28a057",
        #        remote = "https://github.com/eeshakumar/bark-drl",
    )

    _maybe(
        new_git_repository,
        name = "com_github_google_glog",
        commit = "195d416e3b1c8dc06980439f6acd3ebd40b6b820",
        remote = "https://github.com/google/glog",
        build_file = "//:utils/glog.BUILD",
    )

    _maybe(
        #        new_git_repository,
        native.new_local_repository,
        name = "fqn",
        #        branch = "master",
        #        remote = "https://github.com/eeshakumar/fqf-iqn-qrdqn.pytorch",
        path = "/home/ekumar/master_thesis/code/fqf-iqn-qrdqn.pytorch",
        build_file = "//:utils/BUILD.fqn",
    )

    _maybe(
        #        git_repository,
        native.local_repository,
        name = "planner_uct",
        #        branch = "risk_calculation_integration",
        #                commit = "e6434d0f12b8b7b23f29f57df600613a3f294fd2",
        #        remote = "https://github.com/eeshakumar/bark_hypothesis_uct",
        path = "/home/ekumar/master_thesis/code/bark_hypothesis_uct",
    )
