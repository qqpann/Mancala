from mancala.agents.human import HumanAgent
from mancala.agents.random_agent import RandomAgent
from mancala.game import CLIGame
from mancala.mancala import MancalaEnv


def play_cli():
    agents = [HumanAgent(), RandomAgent()]
    env = MancalaEnv()
    game = CLIGame(env, agents)
    game.play_cli()


if __name__ == "__main__":
    play_cli()
