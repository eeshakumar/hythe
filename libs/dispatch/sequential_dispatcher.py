from hythe.libs.dispatch.dispatcher import Dispatcher


class SequentialDispatcher(Dispatcher):

    def __init__(self, dispatch_dict=None):
        super(SequentialDispatcher, self).__init__(dispatch_dict)

    def dispatch(self):
        for (key, value) in self._dispatch_dict.items():
            Dispatcher.execute(dispatch_obj=(key, value))
        return
