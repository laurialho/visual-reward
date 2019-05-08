"""Creates pickle package, which contains demonstrations with split information.
"""
import argparse
import cv2
import numpy as np
import os
import pickle

def averagestdvideo(video):
    """Calculates average standard deviation (std) over video array

    Arguments:
        video {np.array} -- contains images in axis 0

    Returns:
        float -- average std over video
    """
    std = np.std(video, axis=0)
    avg_std = np.average(std)
    return avg_std

def averagestdarray(array):
    """Calculates average standard deviation (std) over std array

    Arguments:
        array {np.array} -- 1-d array

    Returns:
        float -- average std over array
    """
    std = np.std(array)
    return np.average(std)

def split(video, start, end, n, min_size, prev_std):
    """Splits given video into visually similar parts. Adapted from paper:
    Pierre Sermanet, Kelvin Xu, and Sergey Levine,
    Unsupervised perceptual rewards for imitation learning,
    Proceedings of Robotics: Science and Systems, 2017.

    Arguments:
        video {np.array} -- video to find split points
        start {int} -- start index of video for splitting
        end {int} -- last index of video for splitting
        n {int} -- split count
        min_size {int} -- minimum size of split
        prev_std {np.array} -- previous stds

    Returns:
        np.array, np.array -- split points, std of splits
    """
    if n == 1:
        return [], [averagestdvideo(video[start:end])]

    min_std = 0
    min_std_list = []
    min_split = []

    i_start = start + min_size
    i_last = end - ((n - 1) * min_size)
    num = int(i_last - i_start)+1

    indexes = np.linspace(i_start, i_last, num=num)
    indexes = indexes.astype(int)

    for i in indexes:
        std1 = averagestdvideo(video[start:i])

        splits2, std2 = split(video, i, end, n-1, min_size, std1 + prev_std)
        avg_std = averagestdarray([prev_std,std1] + std2)

        if min_std == 0 or avg_std < min_std:
            min_std = avg_std
            min_std_list = np.array(([std1]+std2))
            min_split = [i] + splits2

        print("", end=".")
    return min_split, min_std_list

# Needed arguments:
# hop: for images. Means how many (hop-1) images are skipped.
# splits_n: Defines splits count.
# min_size: mininum split size, -1 if greedy approach.
# base_folder: folder to load demonstrations.
def create_splits(args, base_folder):
    """Creates splits from base folder demonstrations and saves them

    Arguments:
        args {namespace} -- has to contain following arguments:
        hop, splits_n, min_size
        base_folder {string} -- the folder, which contains demonstrations.

    Returns:
        string -- full path to saved pickle file
    """

    sub_folders = os.listdir(base_folder)
    sub_folders.sort()

    # Images are stored here
    images = []
    # Accelerate, steering and braking are stored here
    images_action = []

    for sub_folder in sub_folders:
        # Get files from image folder and sort them
        files = os.listdir(os.path.join(base_folder, sub_folder))
        files.sort()
        # Store images and actions for demonstration
        demonstration = []
        demonstration_action = []
        # Initialize image number
        img_number = 0

        # Load images
        for filename in files:
            # Check is image current file, only then increcement
            if img_number % args.hop == 0 and filename.find(".png") != -1:
                img = cv2.imread(os.path.join(base_folder, sub_folder,filename))
                # Convert to RGB
                if img is not None:
                    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    demonstration.append(RGB_img)
            # Check is a text file
            elif img_number % args.hop == 0 and filename.find(".txt") != -1:
                with open(os.path.join(base_folder, sub_folder,filename), 'r+') as f:
                    for line in f:
                        demonstration_action.append([np.float16(n) \
                            for n in line.strip().split(';')])
                img_number += 1
            elif filename.find(".txt") != -1:
                img_number += 1

        demonstration = np.array(demonstration)
        demonstration_action = np.array(demonstration_action)

        images.append(demonstration)
        images_action.append(demonstration_action)

    #########################
    # Calculate splits for video
    #########################
    min_split = [[] for i in range(len(images))]
    min_std_list = [[] for i in range(len(images))]


    min_size_save_name = args.min_size
    if args.min_size == -1:
        # If faster approach wanted, finds the shortest demonstration and
        # creates min_size based on that:
        args.min_size = \
            int(np.min([len(episode) for episode in images])/args.splits_n)

    # Calculate splits
    for i in range(len(images)):
        print('Starting to calculate splits.')
        print('Demonstration ', i, " started.")
        min_split[i], min_std_list[i] = split(images[i], 0, len(images[i]), \
            args.splits_n, args.min_size, 0)
        print('Demonstration ', i, ' ended')

    ############################
    # Save splits data
    ############################
    save_name = 'imageSplits/split-images%s-min_size%s-splits_n%s-hop%s.pkl' \
        % (len(images),min_size_save_name,args.splits_n,args.hop)

    with open(save_name, 'wb') as f:
        print('Saving data....')
        pickle.dump([images,args.splits_n,min_split,images_action, \
            args.hop, min_size_save_name], f)
        print('Save succesfull: ', save_name)

    return save_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hop', type = int)
    parser.add_argument('-splits_n', type = int)
    parser.add_argument('-min_size', type = int)
    args = parser.parse_args()

    create_splits(args, 'images')
    print('Splits created.')