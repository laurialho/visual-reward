"""Script to create concatenated test image and predicted rewards. Edit the script directly.
"""

from keras.models import load_model
import numpy as np
import pickle
import cv2

# Give full path to imageSplits file
location_splits = r"C:\Users\Lauri\Downloads\CARLA_0.9.4\imageSplits\split-images15-min_size-1-splits_n10-hop5.pkl"
# Give full path to reward model file
location_reward_model = r"C:\Users\Lauri\Downloads\CARLA_0.9.4\rewardModel\reward_model_inceptionv3_imagenet_partial_layers1-18_trained_hop5_splits_n10_min_size-1_epochs80_Adam_lr0.0001.h5"
# Give amount of test demonstrations in the end:
test_count = 3
# Int, bigger than 0
hop_for_predict = 2


print('Predicting and creating image...')

with open(location_splits, 'rb') as f:
    images, splits_n, min_split, images_action, hop, min_size = pickle.load(f)

##########################
# Split videos
##########################
split_video = [[] for i in range(len(images))]
for j in range(len(images)):
    for i in range(splits_n):
        if i == 0:
            split_video[j].insert(i, images[j][0:min_split[j][i]])
        elif i < splits_n-1:
            split_video[j].insert(i, images[j][min_split[j][i-1]:min_split[j][i]])
        else:
            split_video[j].insert(i, images[j][min_split[j][i-1]:(images[j])[:].size])

#####################
# Ravel split_video
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
# Get amount of frames with given test_count
test_frames = 0
if test_count > 0:
    for test in range(test_count):
        test_frames += images_length[-1-test]

X_train, X_test = ravel_splits[0:-test_frames], ravel_splits[-test_frames:]


####################
# Load and predict
####################
model = load_model(location_reward_model)
# Get predictions with defined hop count
prediction = model.predict(X_test[::hop_for_predict])
# Print one long array, split yourself
print(prediction)

# Create long concatenated images from tests
for test in range(test_count):
    test_num = (test+1)*-1

    for i in range(int((len(images[test_num])-1)/hop_for_predict)):
        if i == 0:
            con_image = images[test_num][0]
        else:
            con_image = np.append(con_image, images[test_num][i*hop_for_predict], axis=1)
    cv2.imwrite('testNum%s-predict_hop%s.png' % (test_num, hop_for_predict), \
        cv2.cvtColor(con_image, cv2.COLOR_BGR2RGB))

print('Images created. Please see rewards on top of this. Split yourself')