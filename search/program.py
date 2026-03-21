# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .utils import render_board
from collections import deque
from .check import get_new_possible_states
from heapq import heappush, heappop


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


def heuristic(
    board: dict[Coord, CellState]
) -> int:
    count = 0
    for cell in board.values():
        if cell.color == PlayerColor.BLUE:
            count += 1
    return count
    

def a_star(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    print(render_board(board, ansi=True))

    start = board
    start_key = encode_state(start)

    heap = []
    counter = 0
    
    # push in f = h + g, g, tie_breaker, current board, path
    # tie breaker prevents errors when there are equal priorities  
    heappush(heap, (heuristic(start), 0, counter, start, []))


    # Record the minimum known value of g for each state (the minimum number of steps to reach that state)
    best_g = {start_key: 0}

    while heap:
        # Retrieve the state with the smallest f value from the priority queue 
        f, g, _, current_board, path = heappop(heap)
        current_key = encode_state(current_board)

        # Skip if this record is no longer the optimal solution for the current state.
        if best_g.get(current_key) != g:
            continue

        # Target state found, return action path
        if is_goal(current_board):
            return path
        
        # Expand all successor states of the current state
        for new_possible_state, correct_action in get_new_possible_states(current_board):
            encoded = encode_state(new_possible_state)
            # The actual cost increases by 1 for each action executed
            new_g = g + 1

            # Only keep best path that leads to the better state.
            if encoded not in best_g or new_g < best_g[encoded]:
                best_g[encoded] = new_g
                # Give each new state a unique number to avoid heap comparisons between board objects.
                counter += 1
                new_f = new_g + heuristic(new_possible_state)
                heappush(
                    heap,
                    (new_f, new_g, counter, new_possible_state, path + [correct_action])
                )

    return None


        


