import cv2
import numpy as np
import os

from agent import Agent
from policy_gradient import PolicyGradient
from tools_project import get_file

def run_simulator(args, reward_model_path, action_model_path):
    """Runs the simulator by defining when step is made and a new episode started.

    Arguments:
        args {namespace} -- refer to main.py to see, which args are needed
        reward_model_path {string} -- full path to the reward model
        action_model_path {string} -- full path to the action model
    """

    print('Starting simulator.')

    # Define save path
    save_path = None
    if args.continue_training == 1:
        action_model_path = action_model_path[:-3] + '_trained_gamma%s.h5' % args.gamma
        if args.save_training == 1:
            save_path = action_model_path
    elif args.save_training == 1:
        save_path = action_model_path[:-3] + '_trained_gamma%s.h5' % args.gamma

    PG = PolicyGradient(
        args = args,
        load_path = action_model_path,
        save_path = save_path
    )

    agent = Agent(args,reward_model_path)
    agent.env_start(args)

    # Holds the rewards of the episodes
    rewards = []
    # Holds the current actions
    action = [0.0,0.0,0.0,0.0,0.0]
    # Starting image, saved for new observations
    agent_start_image = None

    # Current step
    step = 0
    # Current episode
    episode = 0

    while episode < args.episodes:
        # Check is game ended by user
        if agent._world is None:
            break
        # Update environment couple of times
        for _ in range(10):
            agent.env_step(action)

        episode += 1

        # Store start image for future episodes
        if agent_start_image is None:
            # Wait that the observation is received
            if Agent._image is None:
                print('Main: Waiting for image')
                while Agent._image is None:
                    pass
            agent_start_image = Agent._image
        else:
            Agent._image = agent_start_image

        # This holds the timer that random action is kept
        random_time = 0
        # How long random action is kept.
        # For example: 20 cycles = 20/60 fps -> 1/3 s random action is kept
        max_random_time = 20
        # If random action is not in use, this increases every step as long
        # that max_random_time is reached. Then zeroed back, so that
        # random action can be tried to use.
        next_random = 0
        # This holds the random action for given time
        random_action = None

        print('New episode started')

        while True:
            step = step+1
            # Store current image
            action = PG.choose_action(Agent._image)

            # Select whether to take random action:
            if args.random_action_prob > 0 and next_random == 0:
                if random_time == 0:
                    # Try to do random action, decrease it after more episodes:
                    if np.random.uniform() < args.random_action_prob-(episode/args.episodes):
                        # Random actions are kept for max_random_time:
                        if random_time == 0:
                            action[0] = np.random.randint(0,2)
                            action[1] = 0
                            action[2] = 0
                            if np.random.randint(1,4) != 1:
                                action[np.random.randint(1,3)] = 1
                            if args.disable_brake == 0:
                                action[-1] = 0 if action[0] == 1 else 1
                            random_action = action
                            random_time = 1
                    else:
                        next_random += 1
                elif random_time < max_random_time:
                    # Replace predicted action with random action, increase timer by 1:
                    action = random_action
                    random_time += 1
                else:
                    # Stop randomness
                    random_time = 0
            elif next_random > 0 and next_random < max_random_time:
                next_random += 1
            else:
                next_random = 0

            # Check game is not quit by user just before next step
            if agent._world is None:
                break
            # Make action, get reward
            reward = agent.env_step(action)

            # If steps count is over max, end episode and add punishment
            if step >= args.steps:
                Agent._done = True
                reward += args.punish_steps

            if args.print_diagnostics == 1:
                if os.path.exists(action_model_path+'_diagnostics.csv') == False:
                    with open(action_model_path+'_diagnostics.csv', 'a') as the_file:
                        the_file.write('episode;step;action;reward;collision;\
                            out_of_area;random_action'+'\n')
                with open(action_model_path+'_diagnostics.csv', 'a') as the_file:
                    the_file.write(str(episode)+';'+str(step)+';'+str(action)\
                        +';'+str(reward)+';'+str(Agent._collision)+';'\
                            +str(Agent._out_of_area)+';'+str(random_time>0)+'\n')
                print('E:%s' % episode, ' S:%s' % step,
                    ' R:%s\t' % np.round(reward,1),
                    ' Col:%s' % Agent._collision,
                    ' Out:%s' % Agent._out_of_area,
                    ' Rnd:%s' % str(random_time>0)[0],
                    ' A:%s' % np.round(action,2))
                if args.save_observations == 1:
                    # Save observations:
                    if not os.path.exists('actionModel/observations'):
                        os.mkdir('actionModel/observations')
                    cv2.imwrite('actionModel/observations/'+str(episode)+\
                        '-'+str(step)+'.png',Agent._image)

            # Store transition for training
            PG.store_transition(Agent._image, action, reward)

            if Agent._done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Sum of episode rewards: ", episode_rewards_sum)

                # Train
                discounted_episode_rewards_norm = PG.learn()


                if os.path.exists(action_model_path+'_rewards.csv') is False:
                    with open(action_model_path+'_rewards.csv', 'a') as the_file:
                        the_file.write('model;episode;episodes;done;collision;\
                            out_of_area;reached_goal;step;steps;\
                                episode_rewards_sum;max_reward_so_far;gamma\n')
                with open(action_model_path+'_rewards.csv', 'a') as the_file:
                    the_file.write(str(action_model_path)+';'+str(episode)+';'\
                        +str(args.episodes)+';'+str(Agent._done)+';'\
                        +str(Agent._collision)+';'+str(Agent._out_of_area)\
                        +';'+str(not(Agent._collision or Agent._out_of_area or step == args.steps))\
                        +';'+str(step)+';'+str(args.steps)+';'\
                        +str(episode_rewards_sum)+';'+str(max_reward_so_far)\
                        +';'+str(args.gamma)+'\n')
                agent.env_restart()
                step = -1
                break
    # Destroy agent
    if agent._world is not None:
        agent.destroy_game()
    del agent
    print('Exit successfull')