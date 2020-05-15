import glob
import os
import re
from typing import Dict, Iterable, Iterator, Tuple

from .exceptions import CannotInterpretTitleError, DuplicateFileTableKeyError, NotExistTableItemError


def _format_title(title: str) -> str:
    return re.sub(r"[\s\-_]+", " ", title).lower()


def _get_title(file_path: str):
    filename: str = os.path.splitext(os.path.split(file_path)[1])[0]
    m = re.match(r"^[\d\s\-_]*$", filename)
    if m:
        return _format_title(m.group(0))
    m = re.match(r"^(CD\d*[\s\-_]*)?\d*[\s\-_]*(.*?)$", filename)
    if m:
        return _format_title(m.group(2))
    raise CannotInterpretTitleError


def _get_files(root_directory: str, file_directories: Iterable[str]) -> Iterator[Tuple[str, str]]:
    files: Dict[str, str] = dict()
    for relative_directory in file_directories:
        for file in glob.glob(root_directory + os.sep + relative_directory, recursive=True):
            try:
                yield _get_title(file), file
                files[_get_title(file)] = file
            except CannotInterpretTitleError:
                pass


def iterate_file_path_set(root_directory: str,
                          key_directories: Iterable[str], item_directories_dict: Dict[str, Iterable[str]],
                          *, key_name: str = "key", raises_error: bool = True) \
        -> Iterator[Tuple[str, Dict[str, str]]]:

    if not key_name or any((k == key_name or not k) for k in item_directories_dict.keys()):
        raise DuplicateFileTableKeyError

    table: Dict[str, Dict[str, str]] = dict()

    # noinspection PyUnusedLocal
    k: str
    # noinspection PyUnusedLocal
    v: str

    for k, v in _get_files(root_directory, key_directories):
        table[k] = dict()
        table[k][key_name] = v

    # noinspection PyUnusedLocal
    item_key: str
    # noinspection PyUnusedLocal
    item_dirs: str

    for item_key, item_dirs in item_directories_dict.items():
        for k, v in _get_files(root_directory, item_dirs):
            if k in table:
                table[k][item_key] = v

    for k, v in table.items():
        for item_key in item_directories_dict.keys():
            if not table[k].get(item_key, ""):
                if raises_error:
                    raise NotExistTableItemError
                else:
                    break
        else:
            yield k, table[k]
