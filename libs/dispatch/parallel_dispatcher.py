from hythe.libs.dispatch.dispatcher import Dispatcher
from torch.multiprocessing import Process, set_start_method


class ParallelDispatcher(Dispatcher):

    def __init__(self, dispatch_list=None):
        super(ParallelDispatcher, self).__init__(dispatch_list)

    def dispatch(self):
        try:
            set_start_method('spawn')
        except RuntimeError as runtime_error:
            print(runtime_error.message)
        for (key, value) in self._dispatch_dict.items():
            print("Dispatched Experiment with seed:", key)
            p = Process(target=self.execute, args=((key, value), ))
            p.start()
            self.experiment_process_list.append(p)
        for p in self.experiment_process_list:
            p.join()
        return
