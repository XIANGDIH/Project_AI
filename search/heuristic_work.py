from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .check import push_stack

def heuristic(board):
    blue_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.BLUE]
    red_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.RED]

    if not blue_stacks:
        return 0.0

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
            t = get_threat(coord_red, state_red, coord_blue, state_blue, board)
            best_threat = min(best_threat, t)

        # The sum of the per-blue nearest-threat distance
        total_dist += best_dist
        # The sum of the per-blue biggest-threat score
        total_threat += best_threat
    
    return len(blue_stacks) + 0.1 * total_dist + 0.5 * total_threat

# Whether the number of enemies has decreased
def has_eliminated (board_prev: dict[Coord, CellState], board_new: dict[Coord, CellState]) -> float:
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


# Get the threat distance between the given Blue and Red pair
def get_threat (coord_red: Coord, state_red: CellState, coord_blue: Coord, state_blue: CellState, board: dict[Coord, CellState]) -> float:

    # EAT possible next step for the given pair
    if next_blue_red(coord_red, coord_blue) and state_red.height >= state_blue.height:
        return 0.1
    
    # CASCADE possible next step for the given pair
    possible_direction = same_direction_get(coord_red, coord_blue)

    if state_red.height >= 2 and possible_direction is not None:
        cascade_weight = successful_cascade_weighted(board, coord_red, state_red, possible_direction)

        if cascade_weight >= 3:
            return 0
        elif cascade_weight > 0:
            return 0.1 * (1/cascade_weight)
        else:
            return 0.3
    
    return 1
# abs(coord_blue.r - coord_red.r) + abs(coord_blue.c - coord_red.c)


# Whether the specific Blue and Red stack pair is next to each other
def next_blue_red (coord_red: Coord, coord_blue: Coord) -> bool:
    distance_r = abs(coord_blue.r-coord_red.r)
    distance_c = abs(coord_blue.c-coord_red.c)
    next_same_r = (distance_r == 0) and (distance_c == 1)
    next_same_c = (distance_c == 0) and (distance_r == 1)
    return next_same_r or next_same_c

# Whether the specific Blue and Red stack pair is in the same direction, if it is get the direction
def same_direction_get (coord_red: Coord, coord_blue: Coord) -> Direction:
    distance_r = abs(coord_blue.r-coord_red.r)
    distance_c = abs(coord_blue.c-coord_red.c)
    if distance_r == 0:
        diff = coord_blue.c-coord_red.c
        if diff > 0:
            return Direction.Right
        else:
            return Direction.Left
    if distance_c == 0:
        diff = coord_blue.r-coord_red.r
        if diff > 0:
            return Direction.Down
        else:
            return Direction.Up
    return None

# Whether the cascade action of the specific Red stack is eliminating enemy stacks
def successful_cascade_weighted (board: dict[Coord, CellState], coord_red: Coord, state_red: CellState, direction: Direction) -> float:
    # Whether the height of the red stack we are looking at is >= 2
    if state_red.height < 2:
        return 0

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
    
    num_eliminated = has_eliminated(board, new_cascaded_board)
    if num_eliminated >= 1:
        return num_eliminated
    
    return 0