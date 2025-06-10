import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

# states: 0-15
# action: 0-left, 1-down, 2-right, 3-left

def run(episodes, is_traning=True):

    env = gym.make('FrozenLake-v1', render_mode='human' if not is_traning else None)

    # define q-learning table
    if is_traning:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("frozen_lake.pkl","rb")
        q = pickle.load(f)
        f.close()

    learning_rate = 0.9
    discounted_factor = 0.9

    epsilon = 1
    epsilon_decay = 0.99
    rng = np.random.default_rng()

    reward_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0
        num_steps = 0

        while (not terminated and not truncated):
            if is_traning and rng.random() < epsilon: #epsilon-greedy algorithm
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            #update q-learning table
            if is_traning:
                q[state, action] = q[state, action] + learning_rate * (reward + discounted_factor * np.max(q[new_state,:]) - q[state, action])
        
            state = new_state
            total_reward += reward
            num_steps += 1

        epsilon = epsilon*epsilon_decay

        if epsilon == 0:
            learning_rate = 0.0001

        if reward == 1:
            reward_per_episode[i] = 1

        print(f"episode: {i}, epsilon: {epsilon}, num_steps: {num_steps}, reward: {reward}")

    env.close()

    if is_traning:
        sum_rewards = np.zeros(episodes)
        for j in range(episodes):
            sum_rewards[j] = np.sum(reward_per_episode[max(0, j-100):(j+1)])
        plt.plot(sum_rewards)
        plt.savefig('frozen_lake.png')

    if is_traning:
        f = open("frozen_lake.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(10000, is_traning=False)
