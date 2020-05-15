from enum import Enum, unique


@unique
class Quality(str, Enum):

    Diminished = "d"
    Minor = "m"
    Perfect = "P"
    Major = "M"
    Augmented = "A"

    # def __str__(self) -> str:
    #     return self.value
