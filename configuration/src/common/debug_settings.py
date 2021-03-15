import sys
import os

cwd = os.getcwd()
workspace_folder = cwd
repo_paths = ["benchmark_database", "com_github_interaction_dataset_interaction_dataset", \
        "com_github_interaction_dataset_interaction_dataset/python", "planner_uct", "hythe", "bark_project", "bark_ml_project"]

executed_file = sys.argv[0]
tmp = executed_file.replace("configuration", "bazel-bin/configuration")
runfiles_dir = tmp.replace(".py", ".runfiles")

sys.path.append(runfiles_dir)
for repo in repo_paths:
    full_path = os.path.join(runfiles_dir, repo)
    print("adding python path: {}".format(full_path))
    sys.path.append(full_path)