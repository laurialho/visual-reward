from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Sequential

def create_partial_model():
    """Creates stripped InceptionV3 network

    Returns:
        string -- full path to the model file
    """

    inp = Input(shape=(299,299,3), name = 'image_input')

    model_base_inception = InceptionV3(weights='imagenet', include_top=False)

    # Get used layers
    first_layer = 1
    last_layer = 18
    layers = model_base_inception.layers[first_layer:last_layer]

    # Create sequential
    partial = Sequential(name='inceptionv3_imagenet_partial')
    for i in range(len(layers)):
        partial.add(layers[i])

    # This is needed, so that partial model can be saved
    model_base_input = partial(inp)
    partial.summary()

    save_name = 'partialModel/' + partial.name + '_layers%s.h5' \
        % (str(first_layer)+'-'+str(last_layer))

    print('Saving model...')
    partial.save(save_name)
    print('Model saved: ', save_name)

    return save_name