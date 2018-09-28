
import os
import numpy as np

def find_class_files(class_label, dir_path, ext="jpg"):
    """
    Find out the files matching the specific class label.

    Parameters
    ----------
    class_label: str
        The class label.

    dir_path: str
        The directory path.

    Returns
    -------
    filenames: list
        The list of filenames.
    """

    filenames = []
    for f in os.listdir(dir_path):
        if f.startswith(class_label) and f.endswith(ext):
            filenames.append(f)

    return filenames

