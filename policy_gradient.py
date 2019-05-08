from keras.models import load_model
import numpy as np
import os
import tensorflow as tf
# Hide unwanted messages for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

class PolicyGradient:
    """Updates the action model with policy gradient
    """

    def __init__(self, args, load_path, save_path = None):
        """Initializer

        Arguments:
            args {namespace} -- arguments, must contain:
            rl_train, gamma, norm_rewards, cut_start, maximize_train_actions
            load_path {string} -- full path to the action model

        Keyword Arguments:
            save_path {string} -- full path to the model save path (default: {None})
        """
        # with tf.device('/device:GPU:0'):
        self.model = load_model(load_path)
        self.save_path = save_path
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        self.train = args.rl_train
        self.gamma = args.gamma
        self.norm_rewards = args.norm_rewards
        self.cut_start = args.cut_start
        self.maximize_train_actions = args.maximize_train_actions

    def store_transition(self, observation, action, reward):
        """Stores transition for training

        Arguments:
            observation {np.array} -- image, shape of action model input
            action {np.array} -- actions probabilities between 0..1
            reward {np.float} -- reward from action in observation
        """

        self.episode_observations.append(observation)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def choose_action(self, observation):
        """Chooses the action for given observation

        Arguments:
            observation {np.array} -- image, shape of action model input

        Returns:
            np.array -- action probabilities between 0..1
        """
        prob_weights = self.model.predict(np.expand_dims(observation,axis=0))
        return prob_weights.ravel()

    def learn(self):
        """Trains the action model

        Returns:
            np.array -- discounted episode rewards
        """
        if self.cut_start == 1:
            cut_length = 140
            print('Cutting first %s steps out from training.' % cut_length)
            self.episode_actions = self.episode_actions[cut_length:]
            self.episode_observations = self.episode_observations[cut_length:]
            self.episode_rewards = self.episode_rewards[cut_length:]
        # Discount episode rewards
        discounted_episode_rewards = self.discount_and_norm_rewards()
        if self.train == True:
            # Train with episode data
            print('Starting to learn')
            if self.maximize_train_actions == 0:
                self.model.fit(x=np.array(self.episode_observations),
                    y=np.array(self.episode_actions),
                    sample_weight=discounted_episode_rewards)
            else:
                print('Converting all train actions to 0 or 1.')
                self.model.fit(x=np.array(self.episode_observations),
                    y=np.round(np.array(self.episode_actions),0),
                    sample_weight=discounted_episode_rewards)

            print('Learning done')

            if self.save_path is not None:
                self.model.save(self.save_path)
                print("Model saved in file: %s" % self.save_path)

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        return discounted_episode_rewards

    def discount_and_norm_rewards(self):
        """Discounts and norms rewards

        Returns:
            np.array -- discounted and normed rewards
        """
        # Adapted from Karpathy: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
        rewards = np.zeros_like(self.episode_rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * self.gamma + self.episode_rewards[t]
            rewards[t] = running_add
        if self.norm_rewards:
            rewards -= np.mean(rewards)
            rewards /= np.std(rewards)
            print('Rewards normed.')
        return rewards


