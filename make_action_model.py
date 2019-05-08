from keras.backend.tensorflow_backend import set_session
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.losses import mean_squared_error, binary_crossentropy
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
import numpy as np
import os
import pickle
import tensorflow as tf
# Hide unwanted messages for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Limit gpu memory if needed
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = set_session(tf.Session(config=config))

def create_action_model(splits_location, partial_model, args):
    """Creates action model. Overwrites old one if with same arguments exists.

    Arguments:
        splits_location {string} -- full path to image splits file (.pkl)
        partial_model {string} -- full path to partial model file (.h5)
        args {namespace} -- has to contain following arguments:
        disable_brake, test_count, action_loss_type, action_lr, action_epochs

    Returns:
        string -- full path to the created action model
    """
    # Load images
    with open(splits_location, 'rb') as f:
        images, splits_n, min_split, images_action, hop, min_size = pickle.load(f)

    ############################################################################
    # Preprocess images
    ############################################################################

    ##########################
    # Split videos and actions
    ##########################
    split_video = [[] for i in range(len(images))]
    split_video_action = [[] for i in range(len(images_action))]

    for j in range(len(images)):
        for i in range(splits_n):
            if i == 0:
                split_video[j].insert(i, images[j][0:min_split[j][i]])
                split_video_action[j].insert(i, images_action[j][0:min_split[j][i]])
            elif i < splits_n-1:
                split_video[j].insert(i, images[j][min_split[j][i-1]:min_split[j][i]])
                split_video_action[j].insert(i, images_action[j][min_split[j][i-1]:min_split[j][i]])
            else:
                split_video[j].insert(i, images[j][min_split[j][i-1]:(images[j])[:].size])
                split_video_action[j].insert(i, images_action[j][min_split[j][i-1]:(images_action[j])[:].size])

    #####################
    # Create one long array from all the images
    #####################
    ravel_splits = []

    for i in range(len(split_video)):
        for j in range(len(split_video[i])):
            for k in range(split_video[i][j][:,1,1,1].size):
                ravel_splits.append(split_video[i][j][k,:,:,:])

    ravel_splits = np.array(ravel_splits)

    ######################
    # Convert actions
    ######################
    # This holds all the actions in one long array.
    y_all = []

    for demonstration in images_action:
        for frame in demonstration:
            y_all.append(frame)

    y_all = np.array(y_all)

    # Steering is saved as -0.5 ... 0.5, +0.5 is needed to get 0..1 range:
    y_all[:,1] += 0.5

    # Default actions count
    actions_count = 3

    # Change the steering to correspond key press.
    for i in range(len(y_all)):
        if y_all[i,1] > 0.5:
            y_all[i,1]  = 1
        elif y_all[i,1] < 0.5:
            y_all[i,1]  = 0

    if args.action_loss_type == 'bce':
        # Expand y_all to have 4 or 5 columns.
        y_all_binary = []

        actions_count = 4
        if actions_count == 4:
            for action in y_all:
                if action[1] == 0.5:
                    y_all_binary.append(np.array((action[0],0,0,action[2])))
                elif action[1] == 0:
                    y_all_binary.append(np.array((action[0],1,0,action[2])))
                elif action[1] == 1:
                    y_all_binary.append(np.array((action[0],0,1,action[2])))

        # Currently depreceated to have also straight action as third column.
        elif actions_count == 5:
            for action in y_all:
                if action[1] == 0.5:
                    y_all_binary.append(np.array((action[0],0,1,0,action[2])))
                elif action[1] == 0:
                    y_all_binary.append(np.array((action[0],1,0,0,action[2])))
                elif action[1] == 1:
                    y_all_binary.append(np.array((action[0],0,0,1,action[2])))

        y_all_binary = np.array(y_all_binary)

        # Replace y_all with binary one
        y_all = y_all_binary

    if args.disable_brake == 1:
        y_all = y_all[:,:-1]
        actions_count -= 1

    #####################
    # Generate train and test splits
    #####################

    # Get demonstration count.
    images_length = [len(i) for i in images]

    # Get amount of images to use as test with given args.test_count
    test_frames = 0
    if args.test_count > 0:
        for test in range(args.test_count):
            test_frames += images_length[-1-test]

    X_train, X_test, y_train, y_test = ravel_splits[0:-test_frames], \
        ravel_splits[-test_frames:], y_all[0:-test_frames], y_all[-test_frames:]

    #############################################################################
    # Action model
    #############################################################################
    inp = Input(shape=(299,299,3), name = 'image_input')

    # with tf.device('/device:GPU:0'):
    partial_trained = load_model(partial_model)
    partial_trained_input = partial_trained(inp)

    partial_trained_input = Conv2D(32,(5,5),activation='relu',padding='same')(partial_trained_input)
    partial_trained_input = MaxPooling2D(pool_size=(4, 4))(partial_trained_input)
    partial_trained_input = Conv2D(32,(5,5),activation='relu',padding='same')(partial_trained_input)
    partial_trained_input = MaxPooling2D(pool_size=(2,2))(partial_trained_input)
    partial_trained_input = Flatten()(partial_trained_input)

    if args.action_loss_type == 'mse':
        partial_trained_input = (Dense(actions_count, activation = 'sigmoid'))(partial_trained_input)
    elif args.action_loss_type == 'bce':
        partial_trained_input = (Dense(actions_count, activation = 'sigmoid'))(partial_trained_input)
    else:
        print('Error! Action model loss type not mse or bce.')
        quit()

    action_model = Model(inputs=inp, outputs=partial_trained_input)
    action_model.summary()

    #####################
    # Loss
    #####################
    loss_2 = None

    if args.action_loss_type == 'bce':
        loss_2 = binary_crossentropy
    elif args.action_loss_type == 'mse':
        loss_2 = mean_squared_error

    #####################
    # Optimizer
    #####################
    optimizer_2 = None

    # Low Learning rate needed
    if args.action_optimizer == 'Adam':
        optimizer_2 = Adam(lr=args.action_lr)
    elif args.action_optimizer == 'sgd':
        optimizer_2 = SGD(lr=args.action_lr)

    #####################
    # Compile
    #####################
    action_model.compile(loss=loss_2, optimizer=optimizer_2,metrics=['accuracy'])

    #####################
    # Fit
    #####################
    last_val_loss = None
    if args.action_epochs > 0:
        history = action_model.fit(X_train, y_train, validation_data = (X_test, y_test))
        epoch = 1
        while last_val_loss != history.history['val_loss'][-1] \
            and epoch < args.action_epochs:
            print('Epoch: %s/%s' % (epoch+1, args.action_epochs))
            last_val_loss = history.history['val_loss'][-1]
            history = action_model.fit(X_train, y_train, \
                validation_data = (X_test, y_test))
            epoch += 1
            if history.history['acc'][-1] > 0.999:
                print('Accuracy over 0.999. Fit ends.')
                break
        if epoch == args.action_epochs:
            print('Maximum number of epochs reached.')
        elif last_val_loss != history.history['val_loss'][-1]:
            print('val_loss was same two times.')

    #####################
    # Save model
    #####################
    action_save_name = 'actionModel\\action_model_'

    action_save_name +=  '%s_%s_lr%s_epochs%s_disable_brake%s_' % \
    (args.action_optimizer, args.action_loss_type, args.action_lr,
    args.action_epochs, args.disable_brake)

    partial_folder = 'partialModel'
    action_save_name += partial_model[len(partial_folder)+1:]

    print('Saving model to: ', action_save_name)
    action_model.save(action_save_name)
    print('Model saved')

    return action_save_name