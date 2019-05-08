"""Does everything, except starts Carla server. Use make_videos.py to create
demonstrations.

This script checks do these files/directories exist in following order:

Demonstration folder: 'images' and 'imageSplits'
Model folders: 'partialModel', 'rewardModel' and 'actionModel'

If any of those doesn't exist, it creates them automatically. After that this
script check does folders contain the files with required arguments in the
following order and requirements:

(1) 'images': At least one directory.

(2) 'imageSplits': file with corresponding arguments: min_size, splits_n, hop

(3) 'partialModel': file without 'trained' in name

(4) 'rewardModel': file with corresponding arguments: same with (2) plus
reward_epochs, reward_optimizer, reward_lr

(5) 'actionModel': file with corresponding arguments: same with (4) plus
action_optimizer, action_epochs, disable_brake, and without 'trained' in name

If some of the files aren't found, they are created automatically. After that,
the Carla client starts.
"""

import argparse
import os

from make_action_model import create_action_model
from make_image_splits import create_splits
from make_partial_model import create_partial_model
from make_reward_model import create_reward_model
from run_simulator import run_simulator
from tools_project import get_file

# Folders to use. DON'T change unless changed in other files too!
folders = ['images','imageSplits','partialModel','rewardModel','actionModel']

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Stores all needed values for training '\
        'and learning.\n Help is: [used_code_file]: [argument definition]')

    # Mainly make_image_splits.py:
    parser.add_argument('-hop', type = int, default = 4,
        help = 'make_image_splits.py: jumps over this many images when images '\
            'are loaded for splitting (default: 4)')
    parser.add_argument('-splits_n', type = int, default = 5,
        help = 'make_image_splits.py: the count of splits used (default: 5)')
    parser.add_argument('-min_size', type = int, default = -1,
        help = 'make_image_splits.py: mininum size of split. '\
            'Set to -1 if greedy and fast approach wanted (default: -1)')

    # Mainly make_reward_model.py:
    parser.add_argument('-test_count', type = int, required=True,
        help = 'make_reward_model.py and make_action_model.py: number of videos\
             used for testing')
    parser.add_argument('-reward_optimizer', type = str, default = 'Adam',
        help = 'make_reward_model.py: optimizer for reward model, Adam or SGD \
            (default: Adam)')
    parser.add_argument('-reward_lr', type = float, default = 0.0001,
        help = 'make_reward_model.py: learning rate for reward model \
            (default: 0.0001)')

    # Mainly make action_model.py:
    parser.add_argument('-reward_epochs', type = int, default = 80,
        help = 'make_action_model.py: maximum number of epochs for training the\
             reward model (default: 80)')
    parser.add_argument('-action_optimizer', type = str, default = 'Adam',
        help = 'make_action_model.py: model optimizer, Adam or SGD (default: Adam)')
    parser.add_argument('-action_loss_type', type = str, default = 'bce',
        help = 'make_action_model.py: model loss type, bce (binary cross \
            entropy) or mse (mean squared error) (default: bce)')
    parser.add_argument('-action_lr', type = float, default = 0.0001,
        help = 'make_action_model.py: model learning rate (default: 0.0001)')
    parser.add_argument('-action_epochs', type = int, default = 40,
        help = 'make_action_model.py: maximum number of epochs for training the\
             action model (default: 40)')
    parser.add_argument('-disable_brake', type = int, default = 1,
        help = 'run_simulator.py: 0 if brake is available action, 1 if \
            disabled always (default: 1)')

    # Mainly policy_gradient.py:
    parser.add_argument('-rl_train', type = int, default = 0,
        help = 'policy_gradient.py: 0 if training is not done for the action \
            model, 1 if is done. (default: 0)')
    parser.add_argument('-gamma', type = float, default = 0.5,
        help = 'policy_gradient.py: used gamma value for discounting episode \
            rewards (default: 0.5)')
    parser.add_argument('-norm_rewards', type = int, default = 1,
        help = 'policy_gradient.py: 0 if discounted episode rewards are not \
            normed, 1 if are (default: 1)')
    parser.add_argument('-save_training', type = int, default = 0,
        help = 'policy_gradient.py: 0 if training is never saved, 1 if is saved \
            after each episode (default: 0)')
    parser.add_argument('-cut_start', type = int, default = 0,
        help = 'policy_gradient.py: 1 if first 140 steps are cut out for \
            training, 0 if not (default: 0)')
    parser.add_argument('-maximize_train_actions', type = int, default = 0,
        help = 'policy_gradient.py: 0 if predicted actions given to rl train \
            as is, 1 if rounded to always 0 or 1 (default: 0)')

    # Mainly run_simulator.py:
    parser.add_argument('-random_action_prob', type = float, default = 0,
        help = 'run_simulator_py: defines how often random action is selected \
            instead of predicted action. Set 0 to never, otherwise between 0..1 (default: 0')
    parser.add_argument('-steps', type = int, default = 800,
        help = 'run_simulator.py: maximum number of steps until episode is \
            ended automatically (default: 800)')
    parser.add_argument('-episodes', type = int, default = 50,
        help = 'run_simulator.py: maximum number of episodes the program runs \
            (default: 50)')
    parser.add_argument('-continue_training', type = int, default = 0,
        help = 'run_simulator.py: 0 if a fresh action model is loaded, 1 if \
            model training is continued (_trained_gamma[].h5 file is the \
                trained model) (default: 0)')
    parser.add_argument('-punish_steps', type = int, default = -100,
        help = 'run_simulator.py: amount to punish, if used steps are more than\
             defined (default: -100)')
    parser.add_argument('-print_diagnostics', type = int, default = 1,
        help = 'run_simulator.py: 0 if no diagnostic of steps print, \
            1 otherwise  (default: 1)')
    # Mainly agent.py:
    parser.add_argument('-user_control', type = int, default = 0,
        help = 'agent.py: 0 if the robot drives the car, 1 if the user drives \
            the car (default: 0)')
    parser.add_argument('-punish_collision', type = int, default = -100,
        help = 'agent.py: amount to punish, if goes out of the defined driving \
            area (default: -100)')
    parser.add_argument('-keep_throttle', type = float, default = -1,
        help = 'agent.py: -1 if no default throttle, otherwise set throttle \
            always between 0...1  (default: -1)')
    parser.add_argument('-reward_goal', type = int, default = 100,
        help = 'agent.py amount to reward, if gets to the goal \
            (default: 100)')
    parser.add_argument('-save_observations', type = int, default = 0,
        help = 'agent.py 0 if observation images are not saved, \
            1 otherwise  (default: 0)')
    parser.add_argument('-new_location', type = int, default = 0,
        help = 'run_simulator.py: 0 if default starting location is used, \
            1 if another location with same trajectory (default: 0)')
    # Carla environment arguments in agent.py
    parser.add_argument('-res',metavar='WIDTHxHEIGHT',default='640x360',
        help='window resolution (default: 640x360)')

    args = parser.parse_args()

    # Model locations
    splits_location = None
    partial_model = None
    reward_model = None
    action_model = None

    print('Checking does every directory exist...', folders)
    for folder in folders:
        os.makedirs(folder,exist_ok=True)
    print('\tDirectories created/exists.\n')

    # Check demonstrations
    print('Checking does ', folders[0], ' folder contain demonstrations.')
    if get_file(folders[0],-1,required_word='out') is None:
        print('\tDemonstrations do not exist. Create them first with \
            make_videos.py file.\n')
        #quit()
    else:
        print('\tDemonstration folders found.\n')

    print('Checking does image splits with given agruments exist...')
    splits_location = get_file(folders[1],-1,required_word=\
        'min_size%s-splits_n%s-hop%s' % (args.min_size,args.splits_n,args.hop))
    if splits_location is None:
        print('\t Not found')
        print('\tCreating image splits with given args.')
        splits_location = create_splits(args,folders[0])
    else:
        print('\tImage splits already created.\n')

    print('Checking does partial model exist...')
    if get_file(folders[2],-1,no_word='trained') is not None:
        print('\tPartial model found.\n')
    else:
        print('\tPartial model not found. Creating one...')
        create_partial_model()
        print('\tPartial model created.\n')

    print('Checking does trained partial and reward models exist...')
    partial_name = 'inceptionv3_imagenet_partial_layers1-18_trained_' +\
        'hop%s_splits_n%s_min_size%s_epochs%s_%s_lr%s.h5' % \
        (args.hop, args.splits_n, args.min_size, args.reward_epochs,
        args.reward_optimizer, args.reward_lr)
    partial_model = get_file(folders[2],-1,required_word=partial_name)
    reward_model = get_file(folders[3],-1,required_word='reward_model_' \
        + partial_name )
    if partial_model is None or reward_model is None:
        print('\tTrained reward or partial model with given splits and epochs \
            not found. Creating those...')
        partial_model, reward_model = create_reward_model(splits_location,args)
        # Check was trained partial model and trained reward model created
        if partial_model is None or reward_model is None:
            print('\t\tmain.py: Error happened in create_reward_model function.')
            quit()
        else:
            print('\tReward model and partial model created.')
    else:
        print('\tReward model and partial model found.\n')

    print('Checking does action model exist...')
    action_model = get_file(folders[4],-1,no_word='.csv',required_word=
    'action_model_%s_%s_lr%s_epochs%s_disable_brake%s_' % (args.action_optimizer,
    args.action_loss_type,
    args.action_lr, args.action_epochs, args.disable_brake) + partial_name)
    if action_model is None:
        print('\tAction model not found. Creating one...')
        action_model = create_action_model(splits_location, partial_model, args)
        print('\tAction model created.\n')
    else:
        print('\tAction model found.\n')

    print('All required models are already found or created! \n\n',
    'Remember to open Carla server from start.lnk!\n\nNow starting client...\n')
    try:
        run_simulator(args,reward_model,action_model)
    finally:
        print('Simulator closed. Bye!')