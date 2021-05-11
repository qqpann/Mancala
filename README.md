# Mancala

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

Play a game with agents

```shell
$ mancala play --player0 human --player1 random
```

Compare each agents and plot their win rates

```shell
$ mancala arena --num_games 200
           p0_random  p0_exact  p0_max
p1_random       41.0      10.0     5.0
p1_exact        81.5      37.5    27.5
p1_max          93.5      51.5    37.0
```

## Algorithms

Mancala is a game with perfect information.
マンカラは完全情報ゲームです。

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
