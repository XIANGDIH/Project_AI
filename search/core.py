# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from dataclasses import dataclass # enable defining classes for storing date
from enum import Enum
from typing import Generator # used as a type of hint for functions that "yield" values

# WARNING: Please *do not* modify any of the code in this file, as this could
#          break things in the submission environment. Failed test cases due to
#          modification of this file will not receive any marks.
#
#          To implement your solution you should modify the `search` function
#          in `program.py` instead, as discussed in the specification.

BOARD_N = 8


@dataclass(frozen=True, slots=True) # objects are immutable after creation; saves memory and can be a bit faster
class Vector2:
    """
    A simple 2D vector "helper" class with basic arithmetic operations
    overloaded for convenience.
    """
    r: int
    c: int

    # a "less than" comparison--comparing vectors first by row then by column (if the row values are equal)
    # returns a boolean
    def __lt__(self, other: 'Vector2') -> bool:
        return (self.r, self.c) < (other.r, other.c)

    # a production function--making the object hashable, so it can be used in dictionary keys
    def __hash__(self) -> int:
        return hash((self.r, self.c))

    # controls how the vector would be printed
    def __str__(self) -> str:
        return f"Vector2({self.r}, {self.c})"

    # allows vector addition/subtraction/negation/multiplication with either: another vector or a direction
    def __add__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r + other.r, self.c + other.c)

    def __sub__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r - other.r, self.c - other.c)

    def __neg__(self) -> 'Vector2':
        return self.__class__(self.r * -1, self.c * -1)

    def __mul__(self, n: int) -> 'Vector2':
        return self.__class__(self.r * n, self.c * n)

    # gives the value of r then c through iterating
    def __iter__(self) -> Generator[int, None, None]:
        yield self.r
        yield self.c


class Direction(Enum):
    """
    An `enum` capturing the four cardinal directions on the board.
    """
    # each direction is represented by a vector
    Down  = Vector2(1, 0)   # move one row down
    Up    = Vector2(-1, 0)  # move one row up
    Left  = Vector2(0, -1)  # move one column left
    Right = Vector2(0, 1)   # move one column right

    # prints with nice arrows
    def __str__(self) -> str:
        return {
            Direction.Down:  "[↓]",
            Direction.Up:    "[↑]",
            Direction.Left:  "[←]",
            Direction.Right: "[→]",
        }[self]

    # gives the value of the direction through iterating
    def __iter__(self):
        return iter(self.value)

    # gives the specific value of either the row or the column you asked
    def __getattribute__(self, __name: str) -> int:
        match __name:
            case "r":
                return self.value.r
            case "c":
                return self.value.c
            case _:
                return super().__getattribute__(__name)


@dataclass(order=True, frozen=True) # automatically gives comparison methods; immutable
# this class is a special version of the vector class for the game board--since not every vector is a valid board coordinate
# it adds bounds checking
# it inherits from the class "Vector2"
class Coord(Vector2):
    """
    A specialisation of the `Vector2` class, representing a coordinate on the
    game board. This class also enforces that the coordinates are within the
    bounds of the game board.
    """

    # runs after the dataclass is created
    # ensures every Coord is inside the board
    def __post_init__(self):
        if not (0 <= self.r < BOARD_N) or not (0 <= self.c < BOARD_N):
            raise ValueError(f"Out-of-bounds coordinate: {self}")

    # prints the coordinate
    def __str__(self):
        return f"{self.r}-{self.c}"

    # adds/subtracts a direction or vector to a coordinate
    # returns a new coordinate
    # if the new coordinate goes off-board, returns false--raises ValueError
    def __add__(self, other: 'Direction|Vector2') -> 'Coord':
        return self.__class__(
            (self.r + other.r),
            (self.c + other.c),
        )

    def __sub__(self, other: 'Direction|Vector2') -> 'Coord':
        return self.__class__(
            (self.r - other.r),
            (self.c - other.c)
        )


class PlayerColor(Enum):
    """
    An `enum` capturing the two player colours.
    """
    RED = 0
    BLUE = 1

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
# represents what is in ONE board cell
class CellState:
    """
    A structure representing the state of a cell on the game board. A cell can
    be empty or contain a stack of tokens of a given player colour and height.
    """
    color: PlayerColor | None = None
    height: int = 0

    # makes sure that the cell's value is valid
    def __post_init__(self):
        if self.color is None and self.height != 0:
            raise ValueError("Empty cell cannot have non-zero height")
        if self.color is not None and self.height <= 0:
            raise ValueError("Stack must have positive height")

    # checks whether the cell is empty or not
    # returns True if the cell is empty
    @property
    def is_empty(self) -> bool:
        return self.color is None
    
    # returns True if the cell contains a stack
    @property
    def is_stack(self) -> bool:
        return self.color is not None

    # prints nicely following the board-style
    def __str__(self):
        if self.is_empty:
            return "."
        color_char = "R" if self.color == PlayerColor.RED else "B"
        return f"{color_char}{self.height}"


@dataclass(frozen=True, slots=True)
class MoveAction:
    """
    A dataclass representing a "move action", which moves a stack one cell
    in a cardinal direction (up, down, left, or right).
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"MOVE({self.coord}, {self.direction})"


@dataclass(frozen=True, slots=True)
class EatAction:
    """
    A dataclass representing an "eat action", which captures an adjacent
    enemy stack. Requires attacker height >= target height.
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"EAT({self.coord}, {self.direction})"


@dataclass(frozen=True, slots=True)
class CascadeAction:
    """
    A dataclass representing a "cascade action", which spreads a stack's tokens
    in a cardinal direction, potentially pushing other stacks.
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"CASCADE({self.coord}, {self.direction})"


Action = MoveAction | EatAction | CascadeAction
