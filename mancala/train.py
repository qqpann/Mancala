from mancala.agents.random import RandomAgent
from mancala.mancala import MancalaEnv


def main():
    env = MancalaEnv(["random", "random"])

    # Try N games
    N = 10
    for i in range(N):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.current_agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            env.state = state

        print(f"Episode {i}: Agent gets {total_reward} reward.")


if __name__ == "__main__":
    main()