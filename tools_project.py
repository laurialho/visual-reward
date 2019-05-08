""" Author: Lauri Alho """

import os
import numpy as np

def get_file(folder, index, include_folder = True, verbose = True, no_word = None, required_word = ''):
    # Get sub folders:
    sub_folders = os.listdir(folder)
    sub_folders.sort()

    file_name = None
    # Return from index
    if len(sub_folders) == 0:
        print('Error, directory is empty! Returning None...')
        return None

    file_name = sub_folders[index]

    if no_word is not None or required_word != '':
        if no_word is None:
            no_word = str(np.random.uniform())
        while file_name.find(no_word) != -1 or file_name.find(required_word) == -1:
            index -= 1
            if index * -1 > len(sub_folders):
                print('Error: File not found without required words! Returning None...')
                return None
            file_name = sub_folders[index]

    if include_folder:
        file_name = os.path.join(folder,file_name)

    if verbose:
        print('From folder: ', folder, ', found file: ', file_name)

    return file_name
