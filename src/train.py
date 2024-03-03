import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

from env_hiv import HIVPatient
from model import DQN, AlwaysPrescribe, ProjectAgent

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

if __name__ == "__main__":
    # envs = gym.vector.AsyncVectorEnv(
    #    [
    #        lambda: TimeLimit(
    #            env=HIVPatient(domain_randomization=False), max_episode_steps=200
    #        )
    #        for _ in range(4)
    #    ]
    # )
    agent = DQN()
    tot_step = 0
    nb_episode = 400
    tot_rewards = []
    tot_loss = []
    pbar = tqdm(range(nb_episode))
    for step in pbar:
        obs, info = env.reset()
        done = False
        truncated = False
        reward_episode = 0
        all_rewards = []
        while not done and not truncated:
            action = agent.act(obs, use_random=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            # Normalize reward: the reward is too big naturally
            all_rewards.append(reward)
            reward = reward / 50_000
            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.learn()
            tot_loss.append(loss)
            reward_episode += reward * 50_000
            tot_step += 1
            obs = next_obs

        pbar.set_description(
            f"Episode {step} done, reward: {np.format_float_scientific(reward_episode , precision=3)}, max reward: {np.format_float_scientific(max(all_rewards) , precision=3)}, loss: {np.format_float_scientific(loss, precision=3)}, epsilon: {np.format_float_scientific(agent.epsilon, precision=3)}"
        )
        tot_rewards.append(reward_episode)
    agent.save("./model.pt")

    print("tot_step: ", tot_step)
    print("tot_rewards: ", tot_rewards)
    plt.plot(tot_rewards)
    plt.show()
    plt.plot(tot_loss)
    plt.show()
