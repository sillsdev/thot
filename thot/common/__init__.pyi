from typing import List, Tuple

class WordAlignmentMatrix():
    def __init__(self, row_length: int, column_length: int) -> None: ...

    @property
    def row_length(self) -> int: ...

    @property
    def column_length(self) -> int: ...
    
    def get_value(self, i: int, j: int) -> int: ...

    def set_value(self, i: int, j: int, value: int) -> None: ...

    def get_asymmetric_alignment(self) -> Tuple[float, List[int]]: ...

__all__ = [
    "WordAlignmentMatrix"
]

