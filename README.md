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
pip install mancala
```

## Usage

```shell
mancala
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
