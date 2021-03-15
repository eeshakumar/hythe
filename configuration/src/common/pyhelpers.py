
import collections
import jsonpickle
import json

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def picklable_object_to_dict(picklable_object):
    dcts = jsonpickle.encode(picklable_object, unpicklable=False, make_refs=False)
    dct = json.loads(dcts)
    return dct