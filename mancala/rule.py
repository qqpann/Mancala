from dataclasses import dataclass


@dataclass
class Rule:
    # Capture opposite stones if the last stone an enters empty pocket.
    capture_opposite: bool = True
    # Continue if the last stone enters the point pocket. other name: multi lap
    continue_on_point: bool = True

    pockets: int = 6
    initial_stones: int = 4
    stones_half: int = 6 * 4