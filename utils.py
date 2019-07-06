import os


def absolute_dir_wrapper(relative_dir):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, relative_dir)
