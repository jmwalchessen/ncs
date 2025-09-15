import os
import sys

def append_directory(recursion_number):

    path_name = __file__
    for i in range(recursion_number):
        path_name = os.path.dirname(path_name)

    return path_name