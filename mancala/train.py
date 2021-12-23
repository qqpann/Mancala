"""Kick off for training A3C agent training"""
import argparse

import torch

from mancala.agents import init_agent
from mancala.agents.a3c.agent import A3CAgent
from mancala.agents.a3c.model import ActorCritic
from mancala.agents.a3c.test import test
from mancala.agents.a3c.train import train
from mancala.agents.random import RandomAgent
from mancala.mancala import MancalaEnv, MancalaState

mp = torch.multiprocessing.get_context("spawn")


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description="A3C for Mancala")
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="learning rate (default: 0.0001)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor for rewards (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=1.00,
    metavar="T",
    help="parameter for GAE (default: 1.00)",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.01,
    metavar="B",
    help="parameter for entropy (default: 0.01)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--num-processes",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=20,
    metavar="NS",
    help="number of forward steps in A3C (default: 20)",
)
parser.add_argument(
    "--max-episode-length",
    type=int,
    default=100,
    metavar="M",
    help="maximum length of an episode (default: 100)",
)
parser.add_argument(
    "--evaluate", action="store_true", help="whether to evaluate results"
)

parser.add_argument(
    "--save-name",
    metavar="FN",
    default="default_model",
    help="path/prefix for the filename to save shared model's parameters",
)
parser.add_argument(
    "--load-name",
    default=None,
    metavar="SN",
    help="path/prefix for the filename to load shared model's parameters",
)


def main():
    agent0 = init_agent("a3c", 0)
    agent1 = init_agent("random", 1)
    env = MancalaEnv(agent0, agent1)

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
    # main()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # agent0 = init_agent("a3c", 0)
    # model_path = "outputs/a3c_2021-07-18T15:27:20.536770_best_39.375"  # a model trained 30 mins with init_random_agent
    model_path = "outputs/a3c_2021-07-18T17:38:20.124428_best_65.625"  # additionally trained 45 mins with init_random_agent
    agent0 = A3CAgent(0, model_path=model_path)
    shared_model = agent0._model
    # shared_model = ActorCritic(state.board.shape[0], env.action_space).type(dtype)
    if args.load_name is not None:
        shared_model.load_state_dict(torch.load(args.load_name))
    shared_model.share_memory()

    # train(1,args,shared_model,dtype)
    processes = []

    # Test
    p = mp.Process(target=test, args=(0, args, shared_model, dtype))
    p.start()
    processes.append(p)

    # Train
    if not args.evaluate:
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, dtype))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()