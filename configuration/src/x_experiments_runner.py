from bark_project.modules.runtime.commons.parameters import ParameterServer

from hythe.libs.experiments.experiment_manager import ExperimentManager

is_load = True


def main():
    if is_load:
        params = ["/home/ekumar/master_thesis/code/hythe/output/experiments/exp.175/params_175_1.json",
                  "/home/ekumar/master_thesis/code/hythe/output/experiments/exp.175/params_175_2.json",
                  "/home/ekumar/master_thesis/code/hythe/output/experiments/exp.175/params_175_3.json",
                  "/home/ekumar/master_thesis/code/hythe/output/experiments/exp.175/params_175_4.json",
                  "/home/ekumar/master_thesis/code/hythe/output/experiments/exp_4d50ff35-8ea2-4cdb-8843-83e9ffe6a742/params_4d50ff35-8ea2-4cdb-8843-83e9ffe6a742_1.json",
                  "/home/ekumar/master_thesis/code/hythe/output/experiments/exp_4d50ff35-8ea2-4cdb-8843-83e9ffe6a742/params_4d50ff35-8ea2-4cdb-8843-83e9ffe6a742_2.json"]
        manager = ExperimentManager(params_files=params)
    else:
        params1 = ParameterServer()
        params2 = ParameterServer()
        params = [params1, params2]
        manager = ExperimentManager(params_list=params)
    manager.run_experiments()
    return


if __name__ == "__main__":
    main()
