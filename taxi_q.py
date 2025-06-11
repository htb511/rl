import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

# link: https://gymnasium.farama.org/environments/toy_text/taxi/
# Taxi-v3 environment
# states: 500 (5x5 grid with 4 passengers and 4 destinations)
# actions: 6 (0-5: south, north, east, west, pickup, dropoff)
# reward: -1 for each step, +20 for successful dropoff, -10 for illegal actions

def run(episodes, is_traning=True):

    env = gym.make('Taxi-v3', render_mode='human' if not is_traning else None)

    # define q-learning table
    if is_traning:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("taxi.pkl","rb")
        q = pickle.load(f)
        f.close()

    learning_rate = 0.9
    discounted_factor = 0.9

    epsilon = 1
    epsilon_decay = 0.0001
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

        reward_per_episode[i] = total_reward        

        epsilon = max(epsilon-epsilon_decay, 0)
        if epsilon == 0:
            learning_rate = 0.0001

        print(f"episode: {i}, epsilon: {epsilon}, num_steps: {num_steps}, total_reward: {total_reward}")

    env.close()

    if is_traning:
        sum_rewards = np.zeros(episodes)
        for j in range(episodes):
            sum_rewards[j] = np.sum(reward_per_episode[max(0, j-100):(j+1)])
        plt.plot(sum_rewards)
        plt.savefig('taxi.png')

    if is_traning:
        f = open("taxi.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    print("Starting training...")
    # Train the model
    run(20000)
    print("Training completed. Now running the trained model...")
    # Run the trained model
    run(10, is_traning=False)
