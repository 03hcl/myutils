import os
from os.path import split, splitext
from typing import Tuple

from .exceptions import FileNameArgumentsError


class FileName:

    @property
    def full_name(self) -> str:
        return self.name_without_ext + os.extsep + self.ext if self.ext else self.name_without_ext

    def __init__(self, name_without_ext: str = "", ext: str = "", *, full_name: str = ""):
        self.name_without_ext: str
        self.ext: str
        if full_name:
            if name_without_ext or ext:
                raise FileNameArgumentsError
            name_pair: Tuple[str, str] = splitext(split(full_name)[1])
            self.name_without_ext = name_pair[0]
            self.ext = name_pair[0].replace(os.extsep, "", count=1)
        else:
            self.name_without_ext = name_without_ext
            self.ext = ext

    def get_full_name(self, prefix: str = "", suffix: str = "") -> str:
        result: str = prefix + self.name_without_ext + suffix
        return result + os.extsep + self.ext if self.ext else result
