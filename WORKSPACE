workspace(name = "hythe")

load("//utils:dependencies.bzl", "hythe_deps")

hythe_deps()

load("@bark_hypothesis_uct_project//util:deps.bzl", "planner_uct_rules_dependencies")

planner_uct_rules_dependencies()

load("@bark_ml_project//utils:dependencies.bzl", "bark_ml_dependencies")

bark_ml_dependencies()

load("@bark_project//tools:deps.bzl", "bark_dependencies")

bark_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()
