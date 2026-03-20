# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .utils import render_board
from collections import deque
from .check import get_new_possible_states


def is_goal(
    board: dict[Coord, CellState]
) -> bool:
    """
    Check whether the current board state is a goal state.
    A board is a goal state if there are no BLUE stacks remaining.
    """
    for cell in board.values():
        if cell.color == PlayerColor.BLUE:
            return False
    return True



def encode_state(
    board: dict[Coord, CellState]
) -> tuple:
    """
    """
    items = []
    for coord, cell in board.items():
        items.append((coord.r, coord.c, cell.color, cell.height))
    items.sort()
    return tuple(items)

def search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to `CellState` instances (each with a `.color` and
            `.height` attribute).

    Returns:
        A list of actions (MoveAction, EatAction, or CascadeAction), or `None`
        if no solution is possible.
    """
    # Check the current board situation
    print(render_board(board, ansi=True))

    visited = {encode_state(board)}
    queue = deque([(board, [])])

    while queue:
        current_board, path = queue.popleft()

        if is_goal(current_board):
            return path

        for new_possible_state, correct_action in get_new_possible_states(current_board):
            encoded = encode_state(new_possible_state)

            if encoded not in visited:
                visited.add(encoded)
                queue.append((new_possible_state, path + [correct_action]))

    return None

