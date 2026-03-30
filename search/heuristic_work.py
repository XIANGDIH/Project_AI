# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade
# This file contains a functions that calculate the heuristic value for the current board configuration.

from .core import CellState, Coord, PlayerColor
from .heuristic import detect_board_state, next_blue_red, get_same_direction, BoardState, successful_cascade

def heuristic(board):
    """
    Compute the heuristic value for a board state.
    Lower values indicate states estimated to be closer to a solution.
    """
    blue_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.BLUE]
    red_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.RED]

    if not blue_stacks:
        return 0.0
    
    dist_weight = 0.1
    threat_weight = 0.5

    # Check the current state of the given board, get the patterns the current board has
    state = detect_board_state(blue_stacks, red_stacks)
    if BoardState.COMPACT_ALIGNMENT in state:
        if BoardState.RED_SCARCITY in state:
            # BEST situation: close + few enemies → kill efficiently
            dist_weight -= 0.06
            threat_weight += 0.065

        else:
            # Close but still many enemies
            dist_weight -= 0.05
            threat_weight += 0.06

    elif BoardState.BLUE_SCATTERED in state:
        if BoardState.RED_SCARCITY in state:
            # Mixed case
            dist_weight += 0.02
            threat_weight += 0.01
        else:
            dist_weight += 0.03
            threat_weight -= 0.04

    elif BoardState.RED_SCARCITY in state:
        dist_weight -= 0.02
        threat_weight += 0.04

    total_dist = 0
    total_threat = 0
    for coord_blue, state_blue in blue_stacks:
        # Get the shortest distance between this specific Blue stack and any Red stacks
        best_dist = float("inf")

        # Get the biggest threat between this specific Blue stack and any Red stacks
        best_threat = float("inf")

        for coord_red, state_red in red_stacks:
            # Distance
            d = abs(coord_blue.r - coord_red.r) + abs(coord_blue.c - coord_red.c)
            best_dist = min(best_dist, d)
            # Threat
            t = get_threat(coord_red, state_red, coord_blue, state_blue, board, state)
            best_threat = min(best_threat, t)

        # The sum of the per-blue nearest distance (smallest manhattan distance value)
        total_dist += best_dist
        # The sum of the per-blue biggest threat (smallest threat value)
        total_threat += best_threat
    
    return len(blue_stacks) + dist_weight * total_dist + threat_weight * total_threat

# Get the threat distance between the given Blue and Red pair--smaller value->greater threat to the current Blue stack
# Improvement: Adjust the threat value according to different situation of the board
def get_threat (coord_red: Coord, state_red: CellState, coord_blue: Coord, state_blue: CellState, board: dict[Coord, CellState], state: list[BoardState]) -> float:
    """
    Estimate how threatening a Red stack is to a specific Blue stack.
    Smaller values represent stronger immediate or near-future threat.
    """
    state_impact_cascade = 0.0
    state_impact_same_direction = 0.0

    if BoardState.COMPACT_ALIGNMENT in state:
        state_impact_cascade = 0.02

    if BoardState.EDGE_CORNER_PRESSURE in state or BoardState.RED_SCARCITY in state or BoardState.BLUE_SCATTERED in state:
        state_impact_same_direction = 0.2

    # EAT possible next step for the given pair
    if next_blue_red(coord_red, coord_blue) and state_red.height >= state_blue.height:
        return 0.1
    
    # CASCADE possible next step for the given pair
    possible_direction = get_same_direction(coord_red, coord_blue)

    if state_red.height >= 2 and possible_direction is not None:

        if successful_cascade(board, coord_red, state_red, coord_blue, possible_direction):
            return 0.1 - state_impact_cascade
        else:
            return 0.3 - state_impact_same_direction
    
    return 1.0
