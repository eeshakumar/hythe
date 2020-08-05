from bark_project.modules.runtime.commons.parameters import ParameterServer
from pathlib import Path
import os
from hythe.libs.experiments.experiment_manager import ExperimentManager

is_load = True


def main():
    print("Experiment server at:", os.getcwd())
    if is_load:
        params = [Path.home().joinpath(".cache/output/experiments/exp_2604641e-1dec-4766-8812-64157bc3328d"
                                       "/params_2604641e-1dec-4766-8812-64157bc3328d_1.json"),
                  Path.home().joinpath(".cache/output/experiments/exp_741a2783-4f2a-41fb-be3f-58232d8da902"
                                       "/params_741a2783-4f2a-41fb-be3f-58232d8da902_1.json")]
        manager = ExperimentManager(params_files=params)
    else:
        params1 = ParameterServer()
        params2 = ParameterServer()
        params = [params1, params2]
        manager = ExperimentManager(params_list=params)
    # manager.run_experiments()
    return


if __name__ == "__main__":
    main()
