# H = enemy_count + 0.2 * setup_distance - 0.5 * eat_bonus - 1.0 * best_cascade_gain - 0.3 * same_line_bonus

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .check import push_stack
from enum import Enum

board_n = 7
detect_bound = 2

from enum import Enum, auto

class BoardState(Enum):
    CAMPACT_ALIGNMENT = auto()       # at least 3 blues are close / line-clustered / fortressed
    EDGE_CORNER_PRESSURE = auto() # at least 2 blues are on edges/corners
    RED_SCARCITY = auto()         # number of blues - number of reds >= 3

    def __str__(self) -> str:
        return self.name


# Calculate the heuristic
def heuristic(board: dict[Coord, CellState]):
    # {Basic.1} The number of Blue stacks
    # BUT the special case of saving effort need to be considered
    blue_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.BLUE] # The length of this stack is the baseline
    red_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.RED]

    if not blue_stacks or not red_stacks:
        return 0.0
    enemy_count = len(blue_stacks)

    # Initializations
    # {Basic.2} The sum of the shortest distance between each Blue stack and any Red stack
    setup_distance = 0.0
    w_dist = 0.2

    # {Bonus}
    # The weight of each bonus point
    w_eat = 0.5
    w_cascade = 1.0
    w_same = 0.3

    eat_bonus = 0.0
    best_cascade_gain_bonus = 0.0
    same_line_bonus = 0.0

    # {Current Environment}
    # Check the board's current state and adjust the weight of the bonus
    state = detect_board_state(blue_stacks, red_stacks)
    if BoardState.CAMPACT_ALIGNMENT in state:
        w_cascade += 0.25

    if BoardState.EDGE_CORNER_PRESSURE in state:
        w_eat += 0.2
        w_same += 0.2

    if BoardState.RED_SCARCITY in state:
        w_same += 0.2
        w_dist -= 0.1

    # Get the bonus marks
    setup_distance = get_setup_distance(blue_stacks, red_stacks)
    eat_bonus = get_immediate_eat_bonus(blue_stacks, red_stacks)
    best_cascade_gain_bonus = get_cascade_bonus(board, blue_stacks, red_stacks)
    same_line_bonus = get_same_line_bonus(blue_stacks, red_stacks)
    
    return enemy_count + 0.2 * setup_distance - eat_bonus - 0.2 * best_cascade_gain_bonus - 0.3 * same_line_bonus

# Helpers
# Check whether the given pair is one the same row/column and the distance between them <= 2
def is_dense(coord_outer: Coord, coord_inner: Coord, detect_bound: int = 2) -> bool:
    if coord_outer == coord_inner:
        return False

    same_line = (coord_outer.r == coord_inner.r) or (coord_outer.c == coord_inner.c)
    if not same_line:
        return False

    distance_r = abs(coord_outer.r - coord_inner.r)
    distance_c = abs(coord_outer.c - coord_inner.c)
    distance = distance_r + distance_c

    return distance <= detect_bound

# Check whether the given coordinate is on one of the edge or one of the corner of the board
def is_pressure(coord: Coord) -> bool:
    return (
        coord.r == 0 or coord.r == board_n or
        coord.c == 0 or coord.c == board_n
    )

# Detect the board's state so as to adjust the weight of each bonus in the heuristic
def detect_board_state(
    blue_stacks: list[tuple[Coord, CellState]],
    red_stacks: list[tuple[Coord, CellState]]
) -> list[BoardState]:
    detected_state: list[BoardState] = []

    # Flag A: compact aligned blues
    dense_pair_count = 0
    for i, (coord_a, _) in enumerate(blue_stacks):
        for j in range(i + 1, len(blue_stacks)):
            coord_b, _ = blue_stacks[j]
            if is_dense(coord_a, coord_b, detect_bound=2):
                dense_pair_count += 1
                if dense_pair_count >= 2:
                    detected_state.append(BoardState.CAMPACT_ALIGNMENT)
                    break
        if dense_pair_count >= 2:
            break

    # Flag B: edge / corner pressure
    pressure_num = 0
    for coord_blue, _ in blue_stacks:
        if is_pressure(coord_blue):
            pressure_num += 1
            if pressure_num >= 2:
                detected_state.append(BoardState.EDGE_CORNER_PRESSURE)
                break

    # Flag C: red scarcity
    if len(blue_stacks) - len(red_stacks) >= 2:
        detected_state.append(BoardState.RED_SCARCITY)

    return detected_state


# {Basic}
def get_Manhattan_distance (coord_red, coord_blue) -> float:
    return abs(coord_blue.r - coord_red.r) + abs(coord_blue.c - coord_red.c)

# Use Manhattan distance as the set-up distance
def get_setup_distance (blue_stacks: list[tuple[Coord, CellState]], red_stacks: list[tuple[Coord, CellState]]) -> float:
    if not blue_stacks or not red_stacks:
        return 0.0

    total_dist = 0.0
    for coord_blue, state_blue in blue_stacks:
        # Get the shortest distance between this specific Blue stack and any Red stacks
        best_dist = float("inf")

        for coord_red, state_red in red_stacks:
            # {Basic.2}: Distance
            d = get_Manhattan_distance(coord_red, coord_blue)
            best_dist = min(best_dist, d)
        
        # The sum of the per-blue nearest-Red stack distance
        total_dist += best_dist
    return total_dist
# Warning: when there is no Red stack

# {Bonus}
# Whether the specific Blue and Red stack pair is next to each other
def next_blue_red(coord_red: Coord, coord_blue: Coord) -> bool:
    dr = abs(coord_blue.r - coord_red.r)
    dc = abs(coord_blue.c - coord_red.c)
    return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

# Whether the new coordinate is off the board
def is_off_board_after(coord_old: Coord, step: int, direction: Direction) -> bool:
    dr, dc = direction.value

    coord_new_r = coord_old.r + dr * step
    coord_new_c = coord_old.c + dc * step
    return not (0 <= coord_new_r <= board_n and 0 <= coord_new_c <= board_n)

# Whether the specific Blue and Red stack pair is in the same direction, if it is get the direction
def get_same_direction(coord_red: Coord, coord_blue: Coord) -> Direction | None:
    if coord_red == coord_blue:
        return None

    if coord_red.r == coord_blue.r:
        return Direction.Right if coord_blue.c > coord_red.c else Direction.Left

    if coord_red.c == coord_blue.c:
        return Direction.Down if coord_blue.r > coord_red.r else Direction.Up

    return None

# {EAT}
# Check how many enemies we can eliminate through EAT on the current board
# BUT! EAT action can only eliminate one Blue stack at most!--so only check whether there is at least
def get_immediate_eat_bonus (blue_stacks: list[tuple[Coord, CellState]], red_stacks: list[tuple[Coord, CellState]]) -> float:
    if not blue_stacks or not red_stacks:
        return 0.0

    for coord_red, state_red in red_stacks:

        for coord_blue, state_blue in blue_stacks:
            if next_blue_red(coord_red, coord_blue) and state_red.height >= state_blue.height:
                return 1.0
    return 0.0

# {SAME DIRECTION}
# Check the number of Blue stacks that have at least one aligned (same direction) red
# [can be improved]
def get_same_line_bonus(
    blue_stacks: list[tuple[Coord, CellState]],
    red_stacks: list[tuple[Coord, CellState]]
) -> float:
    if not blue_stacks or not red_stacks:
        return 0.0

    total_num = 0.0

    for coord_blue, state_blue in blue_stacks:
        for coord_red, state_red in red_stacks:
            same_d = get_same_direction(coord_red, coord_blue)

            if same_d is None:
                continue

            # aligned and red is strong enough to threaten this blue later
            if state_red.height >= state_blue.height:
                total_num += 1
                break

            # aligned and blue is close enough to board edge to be pushed/cascaded off
            if is_off_board_after(coord_blue, state_red.height, same_d):
                total_num += 1
                break

    return total_num

# {CASCADE}
# Check how many enemies we can eliminate through CASCADE on the current board
# And how many meaningful CASECADEs we can perform on the current board
# Since we can only make one CASCADE action in the next step anyways, we will return the largest possible number that we can find for each of the Red stack on the current board
def get_cascade_bonus(
    board: dict[Coord, CellState],
    blue_stacks: list[tuple[Coord, CellState]],
    red_stacks: list[tuple[Coord, CellState]]
) -> float:
    if not blue_stacks or not red_stacks:
        return 0.0

    best_possible_cascade = 0.0

    for coord_red, state_red in red_stacks:
        best_for_this_red = 0.0

        # Only check directions where some blue is aligned with this red
        possible_directions: list[Direction] = []

        for coord_blue, state_blue in blue_stacks:
            same_d = get_same_direction(coord_red, coord_blue)
            if same_d is not None and same_d not in possible_directions:
                possible_directions.append(same_d)

        for direction in possible_directions:
            score = successful_cascade_num_elimination(board, coord_red, state_red, direction)
            best_for_this_red = max(best_for_this_red, score)

        best_possible_cascade = max(best_possible_cascade, best_for_this_red)

    return best_possible_cascade

# Whether the number of enemies has decreased
def count_eliminated_stacks (board_prev: dict[Coord, CellState], board_new: dict[Coord, CellState]) -> float:
    # Find the previous number of enemy stacks on the board
    num_prev = 0
    for cell in board_prev.values():
        if cell.color == PlayerColor.BLUE:
            num_prev += 1

    # Find the new number of enemy stacks on the board
    num_new = 0
    for cell in board_new.values():
        if cell.color == PlayerColor.BLUE:
            num_new += 1

    return num_prev - num_new

def get_possible_eats (blue_stacks: list[tuple[Coord, CellState]], red_stacks: list[tuple[Coord, CellState]]) -> float:
    if not blue_stacks or not red_stacks:
        return 0.0

    total_possible_eats = 0.0

    for coord_red, state_red in red_stacks:

        for coord_blue, state_blue in blue_stacks:
            if next_blue_red(coord_red, coord_blue) and state_red.height >= state_blue.height:
                total_possible_eats += 1
    return total_possible_eats

# The overall distance between each Blue stack and the corresponding closest edge
def total_blue_edge_distance(board):
    total = 0
    for coord, state in board.items():
        if state.color == PlayerColor.BLUE:
            total += min(coord.r, 7 - coord.r, coord.c, 7 - coord.c)
    return total

# Whether the cascade action of the specific Red stack is eliminating enemy stacks
def successful_cascade_num_elimination (board: dict[Coord, CellState], coord_red: Coord, state_red: CellState, direction: Direction) -> float:
    bonus = 0.0

    # Whether the height of the red stack we are looking at is >= 2
    if state_red.height < 2:
        return 0.0

    # Generate the new state after this state (after performing CASCADE of the red stack we are looking at)
    new_cascaded_board = board.copy()
    # s1: Delete the current cell
    new_cascaded_board.pop(coord_red, None)

    # s2: Create new state for 1-current height away cells in this direction with each height of 1
    for s in range(1, state_red.height + 1):
        coord_land_r = coord_red.r + s * direction.r
        coord_land_c = coord_red.c + s * direction.c

        # Whether the current lading cell is out of the boundary
        if not (0 <= coord_land_r < 8 and 0 <= coord_land_c < 8):
            break
        coord_land = Coord(
            coord_land_r,
            coord_land_c
            )
                
        # Whether there is a stack on the on the about-to-land cell
        if coord_land in new_cascaded_board:
            # We need to push forward
            push_stack(new_cascaded_board, coord_land, direction.r, direction.c, 8)
                
        # Now the current landing cell is clear
        new_cascaded_board[coord_land] = CellState(state_red.color, 1)
    
    num_eliminated = count_eliminated_stacks(board, new_cascaded_board)
    # Successful cascade
    if num_eliminated >= 1:
        return num_eliminated
    # Meaningful cascade
    else:
        blue_stacks_old = [(c, s) for c, s in board.items() if s.color == PlayerColor.BLUE]
        red_stacks_old = [(c, s) for c, s in board.items() if s.color == PlayerColor.RED]
        blue_stacks_new = [(c, s) for c, s in new_cascaded_board.items() if s.color == PlayerColor.BLUE]
        red_stacks_new = [(c, s) for c, s in new_cascaded_board.items() if s.color == PlayerColor.RED]

        # Prior1: It will push any Blue stack forward to any edge
        old_edge_dist = total_blue_edge_distance(board)
        new_edge_dist = total_blue_edge_distance(new_cascaded_board)

        # Prior2: It will create more possible eats
        old_eats_num = get_possible_eats(blue_stacks_old, red_stacks_old)
        new_eats_num = get_possible_eats(blue_stacks_new, red_stacks_new)

        # Prior3: It will increase the same-line bonus
        old_same_line = get_same_line_bonus(blue_stacks_old, red_stacks_old)
        new_same_line = get_same_line_bonus(blue_stacks_new, red_stacks_new)

        if new_edge_dist + 1 < old_edge_dist:
            bonus = max(bonus, 0.4)
        if new_eats_num > old_eats_num:
            bonus = max(bonus, 0.3)
        if new_same_line > old_same_line:
            bonus = max(bonus, 0.2)
    return bonus


# Warning: new_cascaded_board = {c: CellState(s.color, s.height) for c, s in board.items()}--about the shallow copy when generating a new possible after-cascaded board
