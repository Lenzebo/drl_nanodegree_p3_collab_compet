from utils import *
from collections import deque

import tqdm  # used for progress bar


def trainAgent(env, brain_name, agent, n_episodes=5000, desired_score=30, n_score_window=100):
    """
    Runs the training of the agent for number of episodes and checks when the environment is considered as solved
    :param agent: 
    :param brain_name: 
    :param n_episodes: 
    :param desired_score: 
    :param n_score_window: 
    :return: 
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=n_score_window)  # last n_score_window scores
    t = tqdm.tqdm(range(1, n_episodes + 1))
    min_episodes = 500
    try:
        for i_episode in t:
            agent.train_step(env, brain_name)

            total_r = playOneEpisode(env, brain_name, agent, train_mode=True)

            score = np.max(total_r) # as indicated by the submission, we are interested in the max of both scores
            scores_window.append(score)  # save most recent score
            scores.append(score)  # s|ave most recent score

            t.set_postfix(score=score, avg_score=np.mean(scores_window))
            if i_episode % n_score_window == 0:
                agent.save('model/model_i_' + str(i_episode) + '.pth')
            if np.mean(scores_window) >= desired_score and i_episode > min_episodes:
                break
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    except KeyboardInterrupt:
        print("Training interrupted....")
    agent.save()
    return scores


def main():
    desired_score = 30
    ma_window = 100

    env, brain_name, num_agents, state_size, action_size = initEnvironment()
    agent = createAgent(action_size, state_size, num_agents)
    scores = trainAgent(env, brain_name, agent, n_episodes=5000, desired_score=desired_score, n_score_window=ma_window)

    plotScores(scores, desired_score, ma_window, show_window=True)

    env.close()


if __name__ == "__main__":
    main()
