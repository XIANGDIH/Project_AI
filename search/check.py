# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade
# This file contains functions
# Whether the current movement is legal or not--[P1]
# Whether the current state has been seen--with the built-in python function, we are able to do it directly in the 
# Whether the final goal has been reached

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor

def push_stack(state: dict[Coord, CellState], coord: Coord, dr: int, dc: int, board_n: int) -> None:
    """
    Push the entire stack at `coord` one step in direction (dr, dc).

    If the next cell is occupied, recursively push that stack first.
    If pushed off the board, the stack is eliminated.
    Mutates `state` in place.
    """
    if coord not in state:
        return

    current_stack = state[coord]

    nr = coord.r + dr
    nc = coord.c + dc

    del state[coord]

    # check bounds BEFORE constructing Coord
    if not (0 <= nr < board_n and 0 <= nc < board_n):
        return

    next_coord = Coord(nr, nc)

    if next_coord in state:
        push_stack(state, next_coord, dr, dc, board_n)

    state[next_coord] = current_stack


def get_new_possible_states(given_state: dict[Coord, CellState]) -> list[tuple[dict[Coord, CellState], Action]]:
    # The list of new states (dictionaries) to be return
    new_possible_state_lst = []

    # Loop through all Red stacks on the board
    # 1: Get all current Red stacks out
    red_cells = []
    for coord, cell_state in given_state.items():
        if cell_state.color == PlayerColor.RED:
            red_cells.append(coord)
    
    # 2: Loop through them one by one
    for red_stack_p in red_cells:
        # Get the current state of the current stack
        red_stack_state = given_state[red_stack_p]

        # Loop through 4 possible directions
        for direction in Direction:
            # 0: Check whether the target cell is out of the board
            # If the target cell is out of the board, we no longer need to explore this direction
            if red_stack_p.r + direction.r < 0 or red_stack_p.r + direction.r > 7 or red_stack_p.c + direction.c < 0 or red_stack_p.c + direction.c > 7:
                continue

            # print(direction, direction.r, direction.c)
            # Get the target cell's coordinates
            target_coord = Coord(
                red_stack_p.r + direction.r,
                red_stack_p.c + direction.c
                )

            # 1: Check the state of the target cell--whether it is empty
            # c1: The target cell is not empty
            if target_coord in given_state:
                target_state_occupied = given_state[target_coord]

                # c1.2: Check whether the stack contained by the target cell is friend or enemy
                # c1.2c1: It's a friend
                if target_state_occupied.color == PlayerColor.RED:
                    # Possible action of MOVE-merge--create a new state
                    new_possible_state = given_state.copy()
                    # s1: Update the target cell
                    # 1.1: Get the previous state of the target cell--target_state_occupied
                    # 1.2: add the current stack's height to the previous height
                    new_possible_state[target_coord] = CellState(target_state_occupied.color, target_state_occupied.height + red_stack_state.height)
                    # s2: Delete the current cell--empty cells are simply absent from the dictionary
                    new_possible_state.pop(red_stack_p, None)
                    
                    action_need = MoveAction(red_stack_p, direction)

                    new_possible_state_lst.append((new_possible_state, action_need))
                        
                # c1.2c2: It's an enemy
                else:
                    # c1.2c2.3: Compare the height of the stack against the enemy
                    if red_stack_state.height >= target_state_occupied.height:
                        # c1.2c2.3c1: Greater than equal to the enemy's stack
                        # Possible action of EAT
                        new_possible_state = given_state.copy()
                        # s1: Replace the enemy's stack with our current stack
                        new_possible_state[target_coord] = CellState(red_stack_state.color, red_stack_state.height)
                        # s2: Delete the current cell
                        new_possible_state.pop(red_stack_p, None)

                        action_need = EatAction(red_stack_p, direction)

                        new_possible_state_lst.append((new_possible_state, action_need))
                    # c1.2c2.3c2: Less than the enemy's stack--no possible action in this case
            # c2: The target cell is empty
            else:
                # Possible action of MOVE-relocate
                new_possible_state = given_state.copy()
                # s1: Create a new state for the target, empty cell
                new_possible_state[target_coord] = CellState(red_stack_state.color, red_stack_state.height)
                # s2: Delete the current cell
                new_possible_state.pop(red_stack_p, None)

                action_need = MoveAction(red_stack_p, direction)

                new_possible_state_lst.append((new_possible_state, action_need))
            # c3: Check whether there exist a reasonable CASCADE
            # Check whether the height of our current stack is at least 2
            # cc1: The height is less than 2--impossible to have a new state here
            if red_stack_state.height < 2:
                continue
            # cc2: The height is greater or equal to 2
            new_possible_state = given_state.copy()
            # sss1: Delete the current cell
            new_possible_state.pop(red_stack_p, None)
            # sss2: Create new state for 1-current height away cells in this direction with eahc height of 1
            for s in range(1, red_stack_state.height + 1):
                coord_land_r = red_stack_p.r + s * direction.r
                coord_land_c = red_stack_p.c + s * direction.c
                # Whether the current lading cell is out of the boundary
                if not (0 <= coord_land_r < 8 and 0 <= coord_land_c < 8):
                    break
                coord_land = Coord(
                    coord_land_r,
                    coord_land_c
                    )
                
                # Whether there is a stack on the on the about-to-land cell
                if coord_land in new_possible_state:
                    # We need to push forward
                    push_stack(new_possible_state, coord_land, direction.r, direction.c, 8)
                
                # Now the current landing cell is clear
                new_possible_state[coord_land] = CellState(red_stack_state.color, 1)

            action_need = CascadeAction(red_stack_p, direction)

            new_possible_state_lst.append((new_possible_state, action_need))

    return new_possible_state_lst


def heuristic_bule_num(
    board: dict[Coord, CellState]
) -> int:
    count = 0
    for cell in board.values():
        if cell.color == PlayerColor.BLUE:
            count += 1
    return count

def manhattan(a, b):
    return abs(a.r - b.r) + abs(a.c - b.c)


def edge_distance(coord):
    return min(coord.r, 7 - coord.r, coord.c, 7 - coord.c)


def fast_heuristic(board):
    reds = []
    blues = []

    for coord, cell in board.items():
        if cell.color == PlayerColor.RED:
            reds.append((coord, cell.height))
        else:
            blues.append((coord, cell.height))

    if not blues:
        return 0
    if not reds:
        return 10**9

    # 先处理高蓝堆
    blues.sort(key=lambda x: -x[1])

    red_used_count = {}
    total = 0

    for blue_coord, blue_height in blues:
        best_cost = float("inf")
        best_red = None

        for red_coord, red_height in reds:
            dist = manhattan(blue_coord, red_coord)
            cost = dist

            # 高度不够，额外罚
            if red_height < blue_height:
                cost += 2 * (blue_height - red_height)

            # 红堆高有优势
            cost -= red_height

            # 能直接吃，强烈偏好
            if dist == 1 and red_height >= blue_height:
                cost -= 6

            # 蓝棋靠边，更容易被cascade处理
            cost -= max(0, 2 - edge_distance(blue_coord))

            # 同一红棋被反复依赖，加拥挤罚分
            cost += 2 * red_used_count.get(red_coord, 0)

            if cost < 0:
                cost = 0

            if cost < best_cost:
                best_cost = cost
                best_red = red_coord

        total += best_cost
        red_used_count[best_red] = red_used_count.get(best_red, 0) + 1

    # 红棋整体战力奖励：最大红堆越高越好
    max_red = max(h for _, h in reds)
    total -= max_red

    if total < 0:
        total = 0

    return total


def heuristic_basic(board):
    reds = []
    blues = []

    for coord, cell in board.items():
        if cell.color == PlayerColor.RED:
            reds.append((coord, cell.height))
        else:
            blues.append((coord, cell.height))

    if not blues:
        return 0
    if not reds:
        return 10**9

    total = 0

    for blue_coord, blue_height in blues:
        best_cost = float("inf")

        for red_coord, red_height in reds:
            dist = manhattan(blue_coord, red_coord)

            # 只保留最基础的一项
            cost = dist - red_height
            if cost < 0:
                cost = 0

            if cost < best_cost:
                best_cost = cost

        total += best_cost

    return total // len(blues)

def heuristic_blue_count_height(board):
    blue_count = 0
    blue_height_sum = 0

    for cell in board.values():
        if cell.color == PlayerColor.BLUE:
            blue_count += 1
            blue_height_sum += cell.height

    return blue_count + blue_height_sum

def board_tiebreak(board):
    blue_height_sum = 0
    max_red_height = 0
    red_count = 0

    for cell in board.values():
        if cell.color == PlayerColor.BLUE:
            blue_height_sum += cell.height
        else:
            red_count += 1
            if cell.height > max_red_height:
                max_red_height = cell.height

    return (
        blue_height_sum,   # 蓝总高度越小越好
        -max_red_height,   # 最大红高度越大越好
        red_count          # 红堆数越少越好
    )