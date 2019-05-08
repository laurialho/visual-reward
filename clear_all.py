#!/usr/bin/env python
"""Clear or cache all listed model folders

This script allows you to easily clear all listed model folders, or to cache
all models by inserting _ to the beginning of the filenames.

Keep in mind, that caching files doesn't automatically exclude them when they
are searched in main script.
"""

import shutil
import os

# Folders to use. DON'T change unless are changed in other scripts also!
folders = ['partialModel','rewardModel','actionModel']

if __name__ == "__main__":
    print('Following directories are used:')
    print(folders)

    inp = input('Do you want to cache all models and image splits by adding _ \
        to the beginning of all model files? [yes/no]\n',
        'Or if you want to clear every folder except \'images\' folder, \
            type \'clear\':\n')

    if inp  == 'yes':
        print('Following directories will be cached:')
        for folder in folders:
            print(os.path.abspath(folder))
            for file in os.listdir(folder):
                print('\t',file)

        if input('Type yes, if you want to cache these\n') == 'yes':
            print('Adding _ to %s files...' % str(folders))
            for folder in folders:
                for file in os.listdir(folder):
                    shutil.move(
                        os.path.join(folder,file),os.path.join(folder,'_'+file))
            print('Adding successfull')
        else:
            print('Adding aborted')

    elif inp == 'clear':
        print('Following directories will be cleared:')
        for folder in folders:
            print(os.path.abspath(folder))
            for file in os.listdir(folder):
                print('\t',file)

        if input('Type yes, if you want to clear these\n') == 'yes':
            print('Clearing %s directories...' % str(folders))
            for folder in folders:
                shutil.rmtree(folder)
            print('Clearing successfull')
        else:
            print('Clearing aborted')

    else:
        print('Aborted')
