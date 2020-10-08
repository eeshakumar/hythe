from functools import wraps
from time import time


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        print(f"Function {func.__name__} took {time() - start:0.4f}s")
        return res
    return wrapper
