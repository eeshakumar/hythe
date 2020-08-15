load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def hythe_deps():
    _maybe(
        native.local_repository,
        #        git_repository,
        name = "bark_ml_project",
        #        branch = "macro_actions_fix",
        #        remote = "https://github.com/eeshakumar/bark-ml",
        path = "/home/ekumar/master_thesis/code/bark-ml",
    )

    #    _maybe(
    #        #        git_repository,
    #        native.local_repository,
    #        name = "bark_project",
    #        path = "/home/ekumar/master_thesis/code/bark",
    #        #        branch = "master",
    #        #        remote = "https://github.com/eeshakumar/bark",
    #    )

    _maybe(
        git_repository,
        name = "bark_project",
        commit = "a79739aab6bd782ed171aea78bbc37b314fc7ba4",
        remote = "https://github.com/bark-simulator/bark",
    )

    _maybe(
        #        git_repository,
        native.local_repository,
        name = "bark_hypothesis_uct_project",
        path = "/home/ekumar/master_thesis/code/bark_hypothesis_uct",
        #        branch = "master",
        #        remote = "https://github.com/eeshakumar/bark",
    )

    _maybe(
        new_git_repository,
        name = "com_github_google_glog",
        commit = "195d416e3b1c8dc06980439f6acd3ebd40b6b820",
        remote = "https://github.com/google/glog",
        build_file = "//:utils/glog.BUILD",
    )

    _maybe(
        native.new_local_repository,
        name = "fqn",
        #branch = "master",
        #remote = "https://github.com/eeshakumar/fqf-iqn-qrdqn.pytorch",
        path = "/home/ekumar/master_thesis/code/fqf-iqn-qrdqn.pytorch",
        build_file = "//:utils/BUILD.fqn",
    )
