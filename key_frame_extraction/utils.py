import json 
from pathlib import Path
import re


def find_file_by_integer(folder: str | Path, number: int) -> Path:
    folder = Path(folder)
    pattern = rf'^{number}( |_)[0-9_ -]*.mkv$'
    match = None
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() == ".mkv":
            res = re.match(pattern, p.name)
            if res:
                match = p

    if match is None:
        raise FileNotFoundError(f"No file found containing {number}")

    return match
