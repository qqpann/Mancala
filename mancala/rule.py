from dataclasses import dataclass


@dataclass
class Rule:
    multi_lap: bool = True
    capture_opposite: bool = True
    continue_on_point: bool = True
    pockets: int = 6
    initial_stones: int = 4
    stones_half: int = 6 * 4