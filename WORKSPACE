workspace(name = "hythe")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//utils:dependencies.bzl", "hythe_deps")

hythe_deps()

# -------- BARK Dependency -------------
git_repository(
    name = "bark_project",
    commit = "a78dd0c08af057cadde62ffede2b8e997f5e435f",
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
    commit = "ee6faee750ec51e72f1d6f3514aae45aa57aaca4",
    remote = "https://github.com/eeshakumar/bark_hypothesis_uct",
    #path="/home/julo/development/bark_hypothesis_uct"
)

load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")

planner_uct_rules_dependencies()
# --------------------------------------------------

# -------- Benchmark Database -----------------------
git_repository(
    name = "benchmark_database",
    commit = "ff6e433ecb7878ebe59996f3994ff67483a7c297",
    remote = "https://github.com/bark-simulator/benchmark-database",
)

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")

benchmark_database_dependencies()

benchmark_database_release()
# --------------------------------------------------

# -------- Bark ML -----------------------
git_repository(
    name = "bark_ml_project",
    commit = "cd1f9f524e8e1c33648b39e1a8284a02a6b68f47",
    remote = "https://github.com/eeshakumar/bark-ml",
    # path = "/home/ekumar/master_thesis/code/bark-ml",
)

# load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
# load("@benchmark_database//load:load.bzl", "benchmark_database_release")

# benchmark_database_dependencies()

# benchmark_database_release()
# --------------------------------------------------

# load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
# load("@benchmark_database//load:load.bzl", "benchmark_database_release")

# benchmark_database_dependencies()

# benchmark_database_release()

# load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")

# planner_uct_rules_dependencies()

# -------- MA MCTS -----------------------

git_repository(
    name = "mamcts_project",
    commit="eccbaf1596a8cc68b0c5ae38dbbaa6cc11827553",
    remote = "https://github.com/juloberno/mamcts"
)

# --------------------------------------------------

# Google or tools for mamcts -----------------------------
load("@mamcts_project//util:deps_or.bzl", "google_or_dependencies")
google_or_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
# Load common dependencies.
protobuf_deps()

