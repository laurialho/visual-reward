from keras import optimizers
from keras.layers import BatchNormalization, Input
from keras.layers.core import Dense, Flatten
from keras.models import load_model, Model
import numpy as np
import pickle

from tools_project import get_file

def create_reward_model(splits_location, args):
    """Creates reward model. Overwrites old one if with same arguments exists.

    Arguments:
        splits_location {string} -- full path to image splits file (.pkl)
        args {namespace} -- has to contain following arguments:
        test_count, reward_loss_type, reward_lr, reward_epochs

    Returns:
        [type] -- [description]
    """
    # Load images
    with open(splits_location, 'rb') as f:
        images, splits_n, min_split, images_action, hop, min_size = pickle.load(f)

    #############################################################################
    # Preprocess images
    #############################################################################

    ##########################
    # Split videos
    ##########################
    split_video = [[] for i in range(len(images))]

    # After this, split_video contains lists of sub steps of videos
    for j in range(len(images)):
        for i in range(splits_n):
            if i == 0:
                split_video[j].insert(i, images[j][0:min_split[j][i]])
            elif i < splits_n-1:
                split_video[j].insert(i, images[j][min_split[j][i-1]:min_split[j][i]])
            else:
                split_video[j].insert(i, images[j][min_split[j][i-1]:(images[j])[:].size])


    #####################
    # Create labels / sub steps / rewards
    #####################
    # y_all contains the sub step numbers of the frames
    y_all = []

    for i in range(len(split_video)):
        for j in range(splits_n):
            for k in range((split_video[i][j])[:,1,1,1].size):
                y_all.append(j)

    y_all = np.array(y_all)

    #####################
    # Create one long array from all the images
    #####################
    ravel_splits = []

    for i in range(len(split_video)):
        for j in range(len(split_video[i])):
            for k in range(split_video[i][j][:,1,1,1].size):
                ravel_splits.append(split_video[i][j][k,:,:,:])

    ravel_splits = np.array(ravel_splits)

    ####################
    # Create splits
    ####################
    # Length of demonstrations
    images_length = [len(i) for i in images]
    # Get amount of frames with given args.test_count
    test_frames = 0
    if args.test_count > 0:
        for test in range(args.test_count):
            test_frames += images_length[-1-test]

    X_train, X_test, y_train, y_test = ravel_splits[0:-test_frames], \
        ravel_splits[-test_frames:], y_all[0:-test_frames], y_all[-test_frames:]

    #############################################################################
    # Reward model
    #############################################################################
    inp = Input(shape=(299,299,3), name = 'image_input')

    # Get model file name (created in make_partial_model.py):
    partial_folder = 'partialModel'
    # Load the last no trained one
    partial_name = get_file(partial_folder,-1,no_word='trained')

    # with tf.device('/device:GPU:0'):
    partial = load_model(partial_name)
    partial_input = partial(inp)
    partial_input = Flatten(name='flatten')(partial_input)
    partial_input = Dense(1)(partial_input)

    reward_model = Model(inputs=inp, outputs=partial_input)
    reward_model.summary()

    ####################
    # Optimizer
    ####################
    optimizer_name = args.reward_optimizer
    # Low learning rate is needed
    if args.reward_optimizer == 'Adam':
        args.reward_optimizer = optimizers.Adam(lr=args.reward_lr)
    elif args.reward_optimizer == 'sgd':
        args.reward_optimizer = optimizers.SGD(lr=args.reward_lr)
    else:
        print('Error! Reward optimizer not defined correctly (Adam or sgd).')
        quit()

    #####################
    # Compile
    #####################
    reward_model.compile(loss='mse',optimizer=args.reward_optimizer,metrics=['accuracy'])

    #####################
    # Fit
    #####################
    last_val_loss = None
    if args.reward_epochs > 0:
        history = reward_model.fit(X_train, y_train, \
            validation_data = (X_test, y_test))
        epoch = 1

        while last_val_loss != history.history['val_loss'][-1] \
            and epoch < args.reward_epochs:
            print('Epoch: %s/%s' % (epoch+1, args.reward_epochs))
            last_val_loss = history.history['val_loss'][-1]
            history = reward_model.fit(X_train, y_train, \
                validation_data = (X_test, y_test))
            epoch += 1
            if history.history['acc'][-1] > 0.999:
                print('Accuracy over 0.999. Fit ends.')
                break
        if epoch == args.reward_epochs:
            print('Maximum number of args.reward_epochs reached.')
        elif last_val_loss != history.history['val_loss'][-1]:
            print('val_loss was same two times.')

    # Freeze learning of partialModel:
    for layer in partial.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    print('Trained partial model is freezed.')

    #####################
    # Save model
    #####################
    # Partial model for action predicting
    partial_save_name = partial_name[:-3] + \
        ('_trained_hop%s_splits_n%s_min_size%s_epochs%s_%s_lr%s' \
            % (hop, splits_n, min_size, args.reward_epochs, optimizer_name, \
                args.reward_lr)) +'.h5'
    partial.save(partial_save_name)
    print('Partial model saved: ', partial_save_name)
    # Reward model
    reward_save_name = 'rewardModel\\reward_model_' + \
        partial_save_name[len(partial_folder)+1:]
    reward_model.save(reward_save_name)
    print('Reward model saved: ', reward_save_name)

    return partial_save_name, reward_save_name




