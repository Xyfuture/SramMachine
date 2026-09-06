"""Common metadata for hardware commands; resource IDs name simulator instances."""
from dataclasses import dataclass


def nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Command:
    cmd_id: str
    op_id: str
    resource_id: str

    def __post_init__(self) -> None:
        for name in ("cmd_id", "op_id", "resource_id"):
            nonempty(name, getattr(self, name))
