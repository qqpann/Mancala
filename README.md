# Mancala

![PyPI - Downloads](https://img.shields.io/pypi/dm/mancala?label=pip%20install%20mancala)

Mancala board game written in python.

![img](https://github.com/qqhann/Mancala/blob/main/assets/preview_cli.png)

## Features & Road maps

- [x] Mancala playable on CLI
- [x] Cmpatible with the gym API
- [ ] Can train RL agents
- [ ] Mancala playable on GUI

## Installation

```shell
$ pip install mancala
```

## Usage

### Play a game with agents

```shell
$ mancala play --player0 human --player1 random
```

### Compare each agents and plot their win rates

The values are player1's (second move) win rates in percentage

```shell
$ mancala arena
              p0_random  p0_exact  p0_max  p0_minimax  p0_negascout
p1_random          50.0      53.0     3.0         0.0           0.0
p1_exact           42.0      48.0     4.0         1.0           1.0
p1_max             95.0      91.0    41.0         0.0           3.0
p1_minimax        100.0      96.0    87.0        30.0          39.0
p1_negascout      100.0      97.0    84.0        19.0          32.0
```

## Algorithms

Mancala is a game with perfect information.
マンカラは完全情報ゲームです。

### Mini-Max

Mini-max is an algorithm for n-player zero-sum games.
The concept is to assume the opponent will take their best move and try to minimize them.

- MiniMax <https://en.wikipedia.org/wiki/Minimax>
- Alpha-beta pruning <https://en.wikipedia.org/wiki/Alpha–beta_pruning>
- Negamax <https://en.wikipedia.org/wiki/Negamax>

### Value Iteration

Using Dynamic Programming (DP), calculate value for states and memorize them.
Use the value to plan future actions.

Other implementations

- OpenSpiel value_iteration
  algorithm <https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/value_iteration.py>
  example <https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/value_iteration.py>

### Policy Iteration

Using Dynamic Programming (DP), calculate value for states and memorize them.
Use the value and policy for planning.

## References

- <https://github.com/mdavolio/mancala>

### Multi agent RL

- <https://github.com/deepmind/open_spiel>
- <https://github.com/PettingZoo-Team/PettingZoo>
