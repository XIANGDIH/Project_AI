# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .utils import render_board
from collections import deque
from .check import get_new_possible_states
from .heuristic_work import heuristic
from heapq import heappush, heappop
import time


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
    Encode board into a canonical key under 8 dihedral symmetries of the 8x8 board.
    This safely merges symmetric states and reduces search blow-up on highly
    symmetric cases (e.g. all-four-corners layouts).
    """
    items: list[tuple[int, int, int, int]] = []
    for coord, cell in board.items():
        items.append((coord.r, coord.c, cell.color.value, cell.height))

    # Build 8 transformed encodings and take the lexicographically smallest one.
    trans = [[] for _ in range(8)]
    for r, c, color, h in items:
        trans[0].append((r, c, color, h))             # identity
        trans[1].append((c, 7 - r, color, h))         # rotate 90
        trans[2].append((7 - r, 7 - c, color, h))     # rotate 180
        trans[3].append((7 - c, r, color, h))         # rotate 270
        trans[4].append((r, 7 - c, color, h))         # mirror vertical
        trans[5].append((7 - r, c, color, h))         # mirror horizontal
        trans[6].append((c, r, color, h))             # main diagonal
        trans[7].append((7 - c, 7 - r, color, h))     # anti-diagonal

    best = None
    for t in trans:
        t.sort()
        key = tuple(t)
        if best is None or key < best:
            best = key

    return best


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
    # - succ_cache: (state_key, prune_move_away) -> ordered list of
    #               (child_key, action, action_rank, child_blue_count)
    state_cache: dict[tuple, dict[Coord, CellState]] = {start_key: start}
    h_cache: dict[tuple, float] = {start_key: heuristic(start)}
    blue_count_cache: dict[tuple, int] = {
        start_key: sum(1 for cell in start.values() if cell.color == PlayerColor.BLUE)
    }
    succ_cache: dict[tuple[tuple, bool], list[tuple[tuple, Action, int, int]]] = {}

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

    def get_successors(
        state_key: tuple,
        prune_move_away: bool = True
    ) -> list[tuple[tuple, Action, int, int]]:
        cache_key = (state_key, prune_move_away)
        cached = succ_cache.get(cache_key)
        if cached is not None:
            return cached

        state_board = state_cache[state_key]
        current_blue_count = get_blue_count(state_key)
        blues_pos = [
            coord for coord, cell in state_board.items()
            if cell.color == PlayerColor.BLUE
        ]
        # Same-layer dedup: keep only the best-ranked edge for each child state.
        best_child: dict[tuple, tuple[int, float, int, Action]] = {}
        for new_possible_state, correct_action in get_new_possible_states(state_board):
            if prune_move_away and isinstance(correct_action, MoveAction) and blues_pos:
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

            candidate = (action_rank, child_h, child_blue_count, correct_action)
            prev = best_child.get(child_key)
            if prev is None or (candidate[0], candidate[1], candidate[2]) < (prev[0], prev[1], prev[2]):
                best_child[child_key] = candidate

        ranked_children: list[tuple[int, float, int, tuple, Action]] = [
            (rank, h, blue_cnt, child_key, action)
            for child_key, (rank, h, blue_cnt, action) in best_child.items()
        ]
        ranked_children.sort(key=lambda x: (x[0], x[1], x[2]))
        ordered_children = [
            (child_key, action, rank, blue_cnt)
            for rank, _, blue_cnt, child_key, action in ranked_children
        ]
        succ_cache[cache_key] = ordered_children
        return ordered_children

    # Heap order:
    # f -> blue_count -> action_rank -> -g -> counter
    # action_rank for start uses 3 (same as MOVE) as neutral default.
    start_blue_count = get_blue_count(start_key)
    heappush(heap, (h_cache[start_key], start_blue_count, 3, 0, counter, 0, start_key))

    # Record the minimum known value of g for each state.
    best_g = {start_key: 0}
    parent = {start_key: None}
    parent_action = {start_key: None}

    best_actions: list[Action] | None = None
    best_len = 10**9

    while heap:
        # Retrieve the state with the smallest f value from the priority queue.
        f, _, _, _, _, g, current_key = heappop(heap)

        # Strict dedup: keep only the best-known g for each state.
        if best_g.get(current_key) != g:
            continue

        current_board = state_cache[current_key]
        expanded += 1

        # Target state found, reconstruct action path.
        if is_goal(current_board):
            best_actions = reconstruct_path(current_key, parent, parent_action)
            best_len = len(best_actions)
            break
        
        # Expand all successor states of the current state.
        parent_key = parent[current_key]
        for encoded, corress_action, action_rank, child_blue_count in get_successors(current_key, prune_move_away=True):
            # Avoid immediate backtracking to parent (safe for shortest paths with unit costs).
            if parent_key is not None and encoded == parent_key:
                continue
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
            heappush(heap, (new_f, child_blue_count, action_rank, -new_g, counter, new_g, encoded))

    if best_actions is None:
        return None

    # Short optimality-improvement phase:
    # g-first search, no move-away pruning, and depth bound < current best.
    phase2_deadline = time.perf_counter() + 6.0
    heap2 = []
    counter += 1
    heappush(heap2, (0, get_h(start_key), counter, start_key))
    best_g2 = {start_key: 0}
    parent2 = {start_key: None}
    parent_action2 = {start_key: None}

    while heap2 and time.perf_counter() < phase2_deadline:
        g, h, _, key = heappop(heap2)
        if best_g2.get(key) != g:
            continue
        if g >= best_len:
            continue

        board_now = state_cache[key]
        if is_goal(board_now):
            better = reconstruct_path(key, parent2, parent_action2)
            if len(better) < best_len:
                best_actions = better
                best_len = len(better)
            break

        parent_key2 = parent2[key]
        for child_key, action, _, _ in get_successors(key, prune_move_away=False):
            if parent_key2 is not None and child_key == parent_key2:
                continue
            ng = g + 1
            if ng >= best_len:
                continue
            old = best_g2.get(child_key)
            if old is not None and ng >= old:
                continue
            best_g2[child_key] = ng
            parent2[child_key] = key
            parent_action2[child_key] = action
            counter += 1
            heappush(heap2, (ng, get_h(child_key), counter, child_key))

    print(f"Generated: {generated}")
    print(f"Expanded: {expanded}")
    return best_actions


        
