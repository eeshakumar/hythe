workspace(name = "hythe")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//utils:dependencies.bzl", "hythe_deps")

hythe_deps()

load("@bark_ml_project//utils:dependencies.bzl", "bark_ml_dependencies")

bark_ml_dependencies()

load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")

planner_uct_rules_dependencies()

load("@bark_project//tools:deps.bzl", "bark_dependencies")

bark_dependencies()

load("@diadem_project//tools:deps.bzl", "diadem_dependencies")

diadem_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# -------- Benchmark Database -----------------------
git_repository(
  name = "benchmark_database",
  commit="ff6e433ecb7878ebe59996f3994ff67483a7c297",
  remote = "https://github.com/bark-simulator/benchmark-database"
)

#local_repository(
#    name = "benchmark_database",
#    path = "/home/ekumar/master_thesis/code/benchmark-database",
#)

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")

benchmark_database_dependencies()

benchmark_database_release()

load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")

planner_uct_rules_dependencies()
