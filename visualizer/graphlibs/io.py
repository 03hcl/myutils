import os


def create_full_path(directory_path: str, file_name: str, extension: str) -> str:
    return directory_path + os.sep + file_name + os.extsep + extension
