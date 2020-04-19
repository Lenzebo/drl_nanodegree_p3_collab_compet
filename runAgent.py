from utils import *


def main():
    env, brain_name, num_agents, state_size, action_size = initEnvironment()
    agent = createAgent(action_size, state_size, num_agents)
    try:
        agent.load()
    except:
        print("Failed to load agent. Running with initial policy")

    scores = playOneEpisode(env, brain_name, agent, verbose=True)

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()


if __name__ == "__main__":
    main()
