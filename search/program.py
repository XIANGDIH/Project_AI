# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .utils import render_board
from collections import deque
from .check import get_new_possible_states
from .heuristic_work import heuristic
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


def _min_dist_to_blue(coord: Coord, blues_list: list[Coord]) -> int:
    best = 10**9
    for blue in blues_list:
        d = abs(coord.r - blue.r) + abs(coord.c - blue.c)
        if d < best:
            best = d
    return best


def search_bfs(
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


def reconstruct_path(goal_key, parent, parent_action):
    actions = []
    current = goal_key

    while parent[current] is not None:
        actions.append(parent_action[current])
        current = parent[current]

    actions.reverse()
    return actions


def search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    print(render_board(board, ansi=True))

    generated = 0   # Total nodes generated
    expanded = 0    # Nodes expanded

    start = board
    start_key = encode_state(start)

    heap = []
    counter = 0

    # Caches:
    # - state_cache: state_key -> canonical board object
    # - h_cache: state_key -> heuristic value
    # - blue_count_cache: state_key -> number of blue stacks
    # - succ_cache: state_key -> ordered list of (child_key, action)
    state_cache: dict[tuple, dict[Coord, CellState]] = {start_key: start}
    h_cache: dict[tuple, float] = {start_key: heuristic(start)}
    blue_count_cache: dict[tuple, int] = {
        start_key: sum(1 for cell in start.values() if cell.color == PlayerColor.BLUE)
    }
    succ_cache: dict[tuple, list[tuple[tuple, Action]]] = {}

    def get_h(state_key: tuple) -> float:
        h = h_cache.get(state_key)
        if h is None:
            h = heuristic(state_cache[state_key])
            h_cache[state_key] = h
        return h

    def get_blue_count(state_key: tuple) -> int:
        count = blue_count_cache.get(state_key)
        if count is None:
            count = sum(
                1 for cell in state_cache[state_key].values()
                if cell.color == PlayerColor.BLUE
            )
            blue_count_cache[state_key] = count
        return count

    def get_successors(state_key: tuple) -> list[tuple[tuple, Action]]:
        cached = succ_cache.get(state_key)
        if cached is not None:
            return cached

        state_board = state_cache[state_key]
        current_blue_count = get_blue_count(state_key)
        blues_pos = [
            coord for coord, cell in state_board.items()
            if cell.color == PlayerColor.BLUE
        ]

        ranked_children: list[tuple[int, float, int, tuple, Action]] = []
        for new_possible_state, correct_action in get_new_possible_states(state_board):
            # Safe pruning: skip relocate-moves that move further away from all blue stacks.
            if isinstance(correct_action, MoveAction) and blues_pos:
                src = correct_action.coord
                dst = src + correct_action.direction
                if dst not in state_board:
                    dist_before = _min_dist_to_blue(src, blues_pos)
                    dist_after = _min_dist_to_blue(dst, blues_pos)
                    if dist_after > dist_before:
                        continue

            child_key = encode_state(new_possible_state)
            if child_key not in state_cache:
                state_cache[child_key] = new_possible_state

            child_blue_count = get_blue_count(child_key)
            child_h = get_h(child_key)

            # Successor ordering:
            # 1) EAT
            # 2) CASCADE that immediately reduces blue count
            # 3) other CASCADE
            # 4) MOVE
            if isinstance(correct_action, EatAction):
                action_rank = 0
            elif isinstance(correct_action, CascadeAction):
                action_rank = 1 if child_blue_count < current_blue_count else 2
            else:
                action_rank = 3

            ranked_children.append(
                (action_rank, child_h, child_blue_count, child_key, correct_action)
            )

        ranked_children.sort(key=lambda x: (x[0], x[1], x[2]))
        ordered_children = [
            (child_key, action)
            for _, _, _, child_key, action in ranked_children
        ]
        succ_cache[state_key] = ordered_children
        return ordered_children

    # Push in f = h + g, g, tie_breaker, state_key
    heappush(heap, (h_cache[start_key], 0, counter, start_key))

    # Record the minimum known value of g for each state.
    best_g = {start_key: 0}
    parent = {start_key: None}
    parent_action = {start_key: None}

    while heap:
        # Retrieve the state with the smallest f value from the priority queue.
        f, g, _, current_key = heappop(heap)

        # Strict dedup: keep only the best-known g for each state.
        if best_g.get(current_key) != g:
            continue

        current_board = state_cache[current_key]
        expanded += 1

        # Target state found, reconstruct action path.
        if is_goal(current_board):
            print(f"Generated: {generated}")
            print(f"Expanded: {expanded}")
            return reconstruct_path(current_key, parent, parent_action)
        
        # Expand all successor states of the current state.
        for encoded, corress_action in get_successors(current_key):
            # The actual cost increases by 1 for each action executed
            new_g = g + 1

            old_g = best_g.get(encoded)
            if old_g is not None and new_g >= old_g:
                continue

            generated += 1
            best_g[encoded] = new_g
            parent[encoded] = current_key
            parent_action[encoded] = corress_action

            # Give each new state a unique number to avoid heap comparison ties.
            counter += 1
            new_f = new_g + get_h(encoded)
            heappush(heap, (new_f, new_g, counter, encoded))

    return None


        

