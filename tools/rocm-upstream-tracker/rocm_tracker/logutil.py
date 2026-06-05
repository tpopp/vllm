from __future__ import annotations

import sys
from typing import TextIO


class Logger:
    def __init__(self, *, verbose: bool = False, stream: TextIO | None = None) -> None:
        self.verbose = verbose
        self.stream = stream or sys.stderr

    def info(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def debug(self, message: str) -> None:
        if self.verbose:
            print(f"[debug] {message}", file=self.stream, flush=True)

    def step(self, message: str) -> None:
        self.info(f"==> {message}")
