import os

def get_root():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.split(root_dir)[0]
