from mancala.agents.random_agent import RandomAgent
from mancala.game import CLIGame
from mancala.mancala import MancalaEnv


def play_cli():
    agent = RandomAgent()
    env = MancalaEnv(agent)
    game = CLIGame(agent, env)
    game.play_cli()


if __name__ == "__main__":
    play_cli()
