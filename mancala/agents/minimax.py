from mancala.state.base import BaseState


def minimax(state: BaseState, depth: int, maximizing_player_id: int) -> float:
    """
    MiniMax function
    """
    # Ref: https://en.wikipedia.org/wiki/Minimax
    if depth == 0 or state.is_terminal():
        return state.rewards_float(maximizing_player_id)

    if state.turn == maximizing_player_id:
        value = -float("inf")
        for act in state.legal_actions(state.turn):
            child = state.clone()
            child.proceed_action(act)
            value = max(value, minimax(child, depth - 1, child.turn))
        return value
    else:
        value = float("inf")
        for act in state.legal_actions(state.turn):
            child = state.clone()
            child.proceed_action(act)
            value = min(value, minimax(child, depth - 1, child.turn))
        return value