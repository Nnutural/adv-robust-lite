from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow({name: row.get(name, "") for name in self.fieldnames})


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
