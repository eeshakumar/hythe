from bark_project.modules.runtime.commons.parameters import ParameterServer
from hythe.libs.experiments.experiment_manager import ExperimentManager


def main():
    params1 = ParameterServer()
    params2 = ParameterServer()
    # params3 = ParameterServer()
    params = [params1, params2]
    manager = ExperimentManager(params_list=params)
    manager.dispatch()
    # experiments = manager.experiments
    # i = 1
    # for (seed, exp) in experiments.items():
    #     print("Starting exp ", i)
    #     exp.run()
    #     i += 1
    return


if __name__ == "__main__":
    main()
