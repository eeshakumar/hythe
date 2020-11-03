workspace(name = "hythe")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//utils:dependencies.bzl", "hythe_deps")

hythe_deps()

# -------- BARK Dependency -------------
git_repository(
    name = "bark_project",
    commit="b9c6b567e7b86fc56df16bd876172904868bc576",
    remote = "https://github.com/juloberno/bark",
)

load("@bark_project//tools:deps.bzl", "bark_dependencies")
bark_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()
#-------------------------------------------

# ------ Planner UCT ------------------------------
git_repository(
  name = "planner_uct",
  commit="d42b19e46e52bd2f7ca0ce92b4376b1ec0507286",
 remote = "https://github.com/juloberno/bark_hypothesis_uct"
  #path="/home/julo/development/bark_hypothesis_uct"
)
load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")
planner_uct_rules_dependencies()
# --------------------------------------------------


# -------- Benchmark Database -----------------------
git_repository(
  name = "benchmark_database",
  commit="ff6e433ecb7878ebe59996f3994ff67483a7c297",
  remote = "https://github.com/bark-simulator/benchmark-database"
)

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")
benchmark_database_dependencies()
benchmark_database_release()
# --------------------------------------------------

# -------- Benchmark Database -----------------------
git_repository(
  name = "bark_ml_project",
  commit = "158f93ecd0c6431788198988f7d02d7aca49c84d",
  remote="https://github.com/bark-simulator/bark-ml",
)

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")
benchmark_database_dependencies()
benchmark_database_release()
# --------------------------------------------------

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")

benchmark_database_dependencies()

benchmark_database_release()

load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")

planner_uct_rules_dependencies()
