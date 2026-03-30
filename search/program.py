# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade
# This file contains a functions that implement our two-phase searching algorithm/strategy.

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor, BOARD_N
from .utils import render_board
from .check import get_new_possible_states
from .heuristic_work import heuristic
from heapq import heappush, heappop
import time

BOARD_LAST_INDEX = BOARD_N - 1
SYMMETRY_TRANSFORMS = 8

# Whether the current state has already eliminated all blue stacks
def is_goal(board: dict[Coord, CellState]) -> bool:
    """
    Check whether the current board state is a goal state.
    A board is a goal state if there are no BLUE stacks remaining.
    """
    for cell in board.values():
        if cell.color == PlayerColor.BLUE:
            return False
    return True

# Convert the board dictionary into a canonical tuple key
# Also fold 8 board symmetries into one key to reduce duplicate states
def encode_state(board: dict[Coord, CellState]) -> tuple:
    """
    Encode board into a canonical key under 8 dihedral symmetries of the board.
    This safely merges symmetric states and reduces search blow-up on highly
    symmetric cases (e.g. all-four-corners layouts).
    """
    # items: list[tuple[int, int, int, int]]
    items = []
    # Flatten dictionary state into sortable tuples
    for coord, cell in board.items():
        items.append((coord.r, coord.c, cell.color.value, cell.height))

    # Build 8 transformed encodings and take the lexicographically smallest one.
    trans = []
    for _ in range(SYMMETRY_TRANSFORMS):
        trans.append([])
    # For each stack, add its coordinate under each symmetry
    for r, c, color, h in items:
        trans[0].append((r, c, color, h))             # identity
        trans[1].append((c, BOARD_LAST_INDEX - r, color, h))         # rotate 90
        trans[2].append((BOARD_LAST_INDEX - r, BOARD_LAST_INDEX - c, color, h))     # rotate 180
        trans[3].append((BOARD_LAST_INDEX - c, r, color, h))         # rotate 270
        trans[4].append((r, BOARD_LAST_INDEX - c, color, h))         # mirror vertical
        trans[5].append((BOARD_LAST_INDEX - r, c, color, h))         # mirror horizontal
        trans[6].append((c, r, color, h))             # main diagonal
        trans[7].append((BOARD_LAST_INDEX - c, BOARD_LAST_INDEX - r, color, h))     # anti-diagonal

    best = None
    # Canonical key = smallest tuple among 8 symmetry variants
    for t in trans:
        t.sort()
        key = tuple(t)
        if best is None or key < best:
            best = key

    return best

# Get the distance from one coordinate to the nearest blue stack
def _min_dist_to_blue(coord: Coord, blues_list: list[Coord]) -> int:
    best = 10**9
    # Scan all blue stacks and keep the minimum Manhattan distance
    for blue in blues_list:
        d = abs(coord.r - blue.r) + abs(coord.c - blue.c)
        if d < best:
            best = d
    return best

# Recover action sequence from parent pointers
def reconstruct_path(goal_key: tuple, parent: dict[tuple, tuple | None], parent_action: dict[tuple, Action | None]) -> list[Action]:
    """
    Reconstruct an action sequence from parent pointers.
    Returns actions in start-to-goal order.
    """
    actions = []
    current = goal_key

    # Follow parent pointers backwards until start state
    while parent[current] is not None:
        actions.append(parent_action[current])
        current = parent[current]

    # Reverse into start -> goal action order
    actions.reverse()
    return actions

class SearchContext:
    """
    Hold per-search caches and helper methods for heuristic and successor expansion.
    """

    def __init__(self, start_key: tuple, start_board: dict[Coord, CellState]):
        # Map encoded state -> board object used by search.
        # self.state_cache: dict[tuple, dict[Coord, CellState]]
        self.state_cache = {start_key: start_board}
        # Map encoded state -> heuristic value.
        # self.h_cache: dict[tuple, float]
        self.h_cache = {start_key: heuristic(start_board)}
        # Map encoded state -> number of blue stacks.
        # self.blue_count_cache: dict[tuple, int]
        start_blue_count = 0
        for cell in start_board.values():
            if cell.color == PlayerColor.BLUE:
                start_blue_count += 1
        self.blue_count_cache = {start_key: start_blue_count}
        # self.succ_cache: dict[tuple[tuple, bool], list[tuple[tuple, Action, int, int]]]
        # Map (encoded state, prune mode) -> ordered successors list.
        self.succ_cache = {}

    # Get heuristic value with cache lookup
    def get_h(self, state_key: tuple) -> float:
        h = self.h_cache.get(state_key)
        if h is None:
            # Compute heuristic only once per encoded state in this search run.
            h = heuristic(self.state_cache[state_key])
            self.h_cache[state_key] = h
        return h

    # Get blue stack count with cache lookup
    def get_blue_count(self, state_key: tuple) -> int:
        count = self.blue_count_cache.get(state_key)
        if count is None:
            # Count BLUE stacks only when this state is first queried.
            count = 0
            for cell in self.state_cache[state_key].values():
                if cell.color == PlayerColor.BLUE:
                    count += 1
            self.blue_count_cache[state_key] = count
        return count

    # Generate successor states, apply optional prune, remove duplicates,
    # then return ordered successors
    def get_successors(self, state_key: tuple, prune_move_away: bool = True) -> list[tuple[tuple, Action, int, int]]:
        cache_key = (state_key, prune_move_away)
        cached = self.succ_cache.get(cache_key)
        if cached is not None:
            # Reuse ordered successors directly if already generated before.
            return cached

        state_board = self.state_cache[state_key]
        current_blue_count = self.get_blue_count(state_key)
        blues_pos = []
        for coord, cell in state_board.items():
            if cell.color == PlayerColor.BLUE:
                blues_pos.append(coord)
        # Keep only one best-ranked edge for each child state
        # best_child: dict[tuple, tuple[int, float, int, Action]]
        best_child = {}
        for new_possible_state, correct_action in get_new_possible_states(state_board):
            # For relocate MOVE, skip options that increase nearest-blue distance
            if prune_move_away and isinstance(correct_action, MoveAction) and blues_pos:
                src = correct_action.coord
                dst = src + correct_action.direction
                if dst not in state_board:
                    dist_before = _min_dist_to_blue(src, blues_pos)
                    dist_after = _min_dist_to_blue(dst, blues_pos)
                    if dist_after > dist_before:
                        continue

            child_key = encode_state(new_possible_state)
            if child_key not in self.state_cache:
                # Register the board object for future cache lookups.
                self.state_cache[child_key] = new_possible_state

            child_blue_count = self.get_blue_count(child_key)
            child_h = self.get_h(child_key)

            # Action priority:
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

        # ranked_children: list[tuple[int, float, int, tuple, Action]]
        ranked_children = []
        for child_key, child_info in best_child.items():
            rank, h, blue_cnt, action = child_info
            ranked_children.append((rank, h, blue_cnt, child_key, action))
        ranked_children.sort(key=lambda x: (x[0], x[1], x[2]))
        ordered_children = []
        for rank, _, blue_cnt, child_key, action in ranked_children:
            ordered_children.append((child_key, action, rank, blue_cnt))
        self.succ_cache[cache_key] = ordered_children
        return ordered_children

# Main search:
# Phase 1 uses heuristic-guided A* ordering
# Phase 2 uses short g-first refinement to try finding a shorter solution
def search(board: dict[Coord, CellState]) -> list[Action] | None:
    """
    Solve the board with a two-stage search strategy.
    Stage 1 finds an initial solution, and stage 2 tries to improve it.
    """
    print(render_board(board, ansi=True))

    # Search counters
    generated = 0
    expanded = 0

    start = board
    start_key = encode_state(start)

    heap = []
    counter = 0

    # Build one context object to hold caches and helper functions for this run.
    ctx = SearchContext(start_key, start)

    # Phase 1 heap order:
    # Pure A*: f = g + h (counter only for stable tie-breaking)
    # Heap entry layout: (f, tie_counter, g, state_key)
    heappush(heap, (ctx.h_cache[start_key], counter, 0, start_key))

    # Record the minimum known g for each state and keep parent links
    best_g = {start_key: 0}
    parent = {start_key: None}
    parent_action = {start_key: None}

    # best_actions: list[Action] | None
    best_actions = None
    best_len = 10**9

    # Phase 1 search loop
    while heap:
        # Pop the best candidate from heap
        f, _, g, current_key = heappop(heap)

        # Skip stale queue entries
        if best_g.get(current_key) != g:
            continue

        current_board = ctx.state_cache[current_key]
        expanded += 1

        # First reached goal in phase 1
        if is_goal(current_board):
            best_actions = reconstruct_path(current_key, parent, parent_action)
            best_len = len(best_actions)
            break
        
        # Expand successors
        parent_key = parent[current_key]
        for encoded, corress_action, _, _ in ctx.get_successors(current_key, prune_move_away=True):
            # Avoid immediate two-step backtracking to parent
            if parent_key is not None and encoded == parent_key:
                continue
            # Unit edge cost
            new_g = g + 1

            old_g = best_g.get(encoded)
            # Keep only strictly better g for each state key
            if old_g is not None and new_g >= old_g:
                continue

            generated += 1
            best_g[encoded] = new_g
            parent[encoded] = current_key
            parent_action[encoded] = corress_action

            # Give each new state a unique number to avoid heap comparison ties.
            counter += 1
            new_f = new_g + ctx.get_h(encoded)
            heappush(heap, (new_f, counter, new_g, encoded))

    # No feasible solution found in phase 1
    if best_actions is None:
        return None

    # Phase 2 refinement:
    # g-first search, no move-away pruning, and depth bound < current best.
    phase2_deadline = time.perf_counter() + 6.0
    heap2 = []
    counter += 1
    # Phase 2 uses g-first order: (g, h, counter, key)
    heappush(heap2, (0, ctx.get_h(start_key), counter, start_key))
    best_g2 = {start_key: 0}
    parent2 = {start_key: None}
    parent_action2 = {start_key: None}

    # Search only for strictly shorter solution in phase 2
    while heap2 and time.perf_counter() < phase2_deadline:
        g, h, _, key = heappop(heap2)
        if best_g2.get(key) != g:
            continue
        # We only care about finding strictly shorter solutions
        if g >= best_len:
            continue

        board_now = ctx.state_cache[key]
        # If this state is already a goal, we found a candidate improvement.
        # Because phase 2 is g-first, this is the shortest goal under current
        # frontier order / pruning constraints.
        if is_goal(board_now):
            better = reconstruct_path(key, parent2, parent_action2)
            if len(better) < best_len:
                best_actions = better
                best_len = len(better)
            break

        parent_key2 = parent2[key]
        # Expand from current state without move-away pruning in phase 2.
        # This allows recovering shorter paths that phase 1 may have missed.
        for child_key, action, _, _ in ctx.get_successors(key, prune_move_away=False):
            if parent_key2 is not None and child_key == parent_key2:
                continue
            ng = g + 1
            # Upper-bound pruning: no need to explore paths that cannot beat
            # the best solution length found so far.
            if ng >= best_len:
                continue
            old = best_g2.get(child_key)
            # Keep only strictly better g for each state in phase 2 as well.
            if old is not None and ng >= old:
                continue
            best_g2[child_key] = ng
            parent2[child_key] = key
            parent_action2[child_key] = action
            counter += 1
            heappush(heap2, (ng, ctx.get_h(child_key), counter, child_key))

    print(f"Generated: {generated}")
    print(f"Expanded: {expanded}")
    return best_actions
