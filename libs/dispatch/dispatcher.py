from abc import abstractmethod, ABC


class Dispatcher(ABC):

    def __init__(self, dispatch_dict=None):
        self._dispatch_dict = dispatch_dict
        self._process_list = []
        return

    def set_dispatch_dict(self, dispatch_dict):
        self._dispatch_dict = dispatch_dict

    @abstractmethod
    def dispatch(self):
        raise NotImplementedError

    @staticmethod
    def execute(dispatch_obj):
        try:
            print("Running experiment with seed:", dispatch_obj[0])
            dispatch_obj[1].run()
        except IOError as io_error:
            print(io_error.message)


