# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade
# This file contains a functions that help to build the logics in the heuristic function.

# H = enemy_count + 0.2 * setup_distance - 0.5 * eat_bonus - 1.0 * best_cascade_gain - 0.3 * same_line_bonus

from .core import CellState, Coord, Direction
from enum import Enum

board_n = 7
detect_bound = 2
detect_bound_dist = 4

from enum import Enum, auto

class BoardState(Enum):
    COMPACT_ALIGNMENT = auto()       # at least 3 Blues are close / line-clustered / fortressed
    EDGE_CORNER_PRESSURE = auto() # at least 2 Blues are on edges/corners
    RED_SCARCITY = auto()         # number of Blues - number of reds >= 3
    BLUE_SCATTERED = auto()   # number of Blues are far away from each other >= 2

    def __str__(self) -> str:
        return self.name

# Helpers
# Check whether the given pair is one the same row/column and the distance between them <= 2
def is_dense (coord_outer: Coord, coord_inner: Coord) -> bool:
    """
    Check whether two coordinates are aligned and close.
    Returns True when they share a row/column within the dense distance bound.
    """
    if coord_outer == coord_inner:
        return False

    same_line = (coord_outer.r == coord_inner.r) or (coord_outer.c == coord_inner.c)
    if not same_line:
        return False

    distance_r = abs(coord_outer.r - coord_inner.r)
    distance_c = abs(coord_outer.c - coord_inner.c)
    distance = distance_r + distance_c

    return distance <= detect_bound

# Check whether the given pair is at least 4 cells from each other
def is_scatter (coord_outer: Coord, coord_inner: Coord) -> bool:
    """
    Check whether two coordinates are far apart.
    Returns True when Manhattan distance is at least the scatter bound.
    """
    if coord_outer == coord_inner:
        return False
    
    return get_Manhattan_distance(coord_inner, coord_outer) >= detect_bound_dist

# Check whether there is a Red stack in the middle of the "dense" Blue stack pairs
def no_red_between (
    coord_a: Coord,
    coord_b: Coord,
    red_stacks: list[tuple[Coord, CellState]]
) -> bool:
    """
    Check whether no red stack lies strictly between two aligned coordinates.
    Returns False for non-aligned coordinates.
    """
    # Must be aligned first
    if coord_a.r == coord_b.r:
        row = coord_a.r
        c_min = min(coord_a.c, coord_b.c)
        c_max = max(coord_a.c, coord_b.c)

        for coord_red, _ in red_stacks:
            if coord_red.r == row and c_min < coord_red.c < c_max:
                return False
        return True

    if coord_a.c == coord_b.c:
        col = coord_a.c
        r_min = min(coord_a.r, coord_b.r)
        r_max = max(coord_a.r, coord_b.r)

        for coord_red, _ in red_stacks:
            if coord_red.c == col and r_min < coord_red.r < r_max:
                return False
        return True

    return False

# Check whether the given coordinate is on one of the edge or one of the corner of the board
def is_pressure (coord: Coord) -> bool:
    """
    Check whether a coordinate is on the board edge or corner.
    """
    return (
        coord.r == 0 or coord.r == board_n or
        coord.c == 0 or coord.c == board_n
    )

def detect_board_state(
    blue_stacks: list[tuple[Coord, CellState]],
    red_stacks: list[tuple[Coord, CellState]]
) -> list[BoardState]:
    """
    Detect high-level board patterns used to adapt heuristic weights.
    """
    detected_state: list[BoardState] = []

    # Flag A: Compact aligned blues with no red in between
    dense_pair_count = 0
    for i, (coord_a, _) in enumerate(blue_stacks):
        for j in range(i + 1, len(blue_stacks)):
            coord_b, _ = blue_stacks[j]

            if is_dense(coord_a, coord_b) and no_red_between(coord_a, coord_b, red_stacks):
                dense_pair_count += 1
                if dense_pair_count >= 2:
                    break

        if dense_pair_count >= 2:
            detected_state.append(BoardState.COMPACT_ALIGNMENT)
            break

    # Flag B: Edge / corner pressure
    pressure_num = 0
    for coord_blue, _ in blue_stacks:
        if is_pressure(coord_blue):
            pressure_num += 1
            if pressure_num >= 2:
                detected_state.append(BoardState.EDGE_CORNER_PRESSURE)
                break

    # Flag C: Red scarcity
    if len(blue_stacks) - len(red_stacks) >= 2:
        detected_state.append(BoardState.RED_SCARCITY)

    # Flag D: Blue scattered
    scatter_pair_count = 0
    for i, (coord_a, _) in enumerate(blue_stacks):
        for j in range(i + 1, len(blue_stacks)):
            coord_b, _ = blue_stacks[j]

            if is_scatter(coord_a, coord_b):
                scatter_pair_count += 1
                if scatter_pair_count >= 2:
                    break

        if scatter_pair_count >= 2:
            break

    if scatter_pair_count >= 2 and dense_pair_count == 0:
        detected_state.append(BoardState.BLUE_SCATTERED)

    return detected_state


# {Basic}
def get_Manhattan_distance (coord_a, coord_b) -> float:
    """
    Compute Manhattan distance between two coordinates.
    """
    return abs(coord_b.r - coord_a.r) + abs(coord_b.c - coord_a.c)

# {Bonus}
# Whether the specific Blue and Red stack pair is next to each other
def next_blue_red (coord_red: Coord, coord_blue: Coord) -> bool:
    """
    Check whether a Red and Blue stack are adjacent orthogonally.
    """
    dr = abs(coord_blue.r - coord_red.r)
    dc = abs(coord_blue.c - coord_red.c)
    return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

# Whether the new coordinate is off the board
def is_off_board_after (coord_old: Coord, step: int, direction: Direction, stack_in_between_num: int) -> bool:
    """
    Check whether moving a coordinate by a number of steps exits the board.
    """
    dr, dc = direction.value

    coord_new_r = coord_old.r + dr * step
    coord_new_c = coord_old.c + dc * step
    return not (0 <= coord_new_r <= board_n and 0 <= coord_new_c <= board_n)

# Whether the specific Blue and Red stack pair is in the same direction, if it is get the direction
def get_same_direction (coord_red: Coord, coord_blue: Coord) -> Direction | None:
    """
    Get the straight-line direction from Red to Blue when aligned.
    Return None when they are not on the same row/column.
    """
    if coord_red == coord_blue:
        return None

    if coord_red.r == coord_blue.r:
        return Direction.Right if coord_blue.c > coord_red.c else Direction.Left

    if coord_red.c == coord_blue.c:
        return Direction.Down if coord_blue.r > coord_red.r else Direction.Up

    return None

# Count how many stacks (no matter Blue or Red) are in between of the given pair
def count_stacks_between (coord_a: Coord, coord_b: Coord, occupied: set[Coord]) -> int:
    """
    Count occupied cells strictly between two aligned coordinates.
    Returns 0 when coordinates are not aligned.
    """
    # Same row
    if coord_a.r == coord_b.r:
        row = coord_a.r
        c_min = min(coord_a.c, coord_b.c)
        c_max = max(coord_a.c, coord_b.c)

        count = 0
        for c in range(c_min + 1, c_max):
            if Coord(row, c) in occupied:
                count += 1
        return count

    # Same column
    if coord_a.c == coord_b.c:
        col = coord_a.c
        r_min = min(coord_a.r, coord_b.r)
        r_max = max(coord_a.r, coord_b.r)

        count = 0
        for r in range(r_min + 1, r_max):
            if Coord(r, col) in occupied:
                count += 1
        return count

    return 0

# Whether the cascade action of the specific Red stack is successful (it can eliminate the corresponding Blue stack we are looking at) or otherwise meaningful (in at least one of the three cases)
# in terms of the given pair
def successful_cascade (board: dict[Coord, CellState], coord_red: Coord, state_red: CellState, coord_blue: Coord, direction: Direction) -> float:
    """
    Check whether a cascade can push the target Blue stack off board.
    """
    is_successful = False

    # Whether the height of the red stack we are looking at is >= 2
    if state_red.height < 2:
        return 0.0
    
    step = state_red.height
    board_stack_on = set(board.keys())
    stack_inbetween_num = count_stacks_between(coord_red, coord_blue, board_stack_on)

    # Get the new position of Blue stack we are looking at 
    if is_off_board_after(coord_blue, step, direction, stack_inbetween_num):
        is_successful = True

    return is_successful
