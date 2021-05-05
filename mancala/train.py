from mancala.agents.random_agent import RandomAgent
from mancala.mancala import MancalaEnv


def main():
    env = MancalaEnv()
    agent = RandomAgent()

    # Try 1 game
    for i in range(10):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            env.state = state

        print(f"Episode {1}: Agent gets {total_reward} reward.")


if __name__ == "__main__":
    main()