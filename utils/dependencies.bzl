load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def _maybe(repo_rule, name, **kwargs):
  # if name not in native.existing_rules():
  repo_rule(name = name, **kwargs)

def hythe_deps():

  _maybe(
    native.local_repository,
    name = "bark_ml_project",
    path = "/home/eeshakumar/school/master_thesis/code/bark-ml",
    #repo_mapping = {"@bark_ml": "@bark_ml//bark_ml"}
  )

  _maybe(
    native.local_repository,
    name = "bark_project",
    path = "/home/eeshakumar/school/master_thesis/code/bark"
  )

  _maybe(
    new_git_repository,
    name = "com_github_google_glog",
    commit = "195d416e3b1c8dc06980439f6acd3ebd40b6b820",
    remote = "https://github.com/google/glog",
    build_file="//:utils/glog.BUILD"
  )

  _maybe(
    native.new_local_repository,
    name = "fqn",
    path = "/home/eeshakumar/school/master_thesis/code/fqf-iqn-qrdqn.pytorch",
    build_file = "//:utils/BUILD.fqn"
  )
