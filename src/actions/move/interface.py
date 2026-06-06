from dataclasses import dataclass

from actions.base import Interface


@dataclass
class MoveInput:
    """
    Input for the Move action.

    Parameters
    ----------
    action : str
        Free-form natural-language sub-task prompt (Korean / English),
        e.g. ``"Walk to the refrigerator and face the door."``.
        Whole-body VLA has no finite action vocabulary — the
        MoveConnector forwards the string to the VLA Provider as-is.
        (Replaces OM1's MovementAction enum, which encoded the old
        Unitree-dog template — STAND_STILL / SIT / DANCE / ...; gone
        with the VLA scope change.)
    """

    action: str


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """Action interface for robot movement commands (whole-body VLA prompt)."""

    input: MoveInput
    output: MoveInput
