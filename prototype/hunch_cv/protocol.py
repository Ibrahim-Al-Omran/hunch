from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    code: str
    name: str
    seconds: float
    prompt: str


PROTOCOL: tuple[Step, ...] = (
    Step("A", "calibration_upright", 10, "Sit naturally upright and look at your screen."),
    Step("A2", "calibration_slouch", 8, "Slouch the way you do when tired: head forward, shoulders rounded. Hold."),
    Step("B", "upright_typing", 45, "Sit upright and work normally (type, use the mouse)."),
    Step("C", "mild_forward_head", 30, "Back upright, but push your head slightly forward (3-5 cm)."),
    Step("D", "forward_head", 30, "Push your head clearly forward toward the screen."),
    Step("E", "slouch", 30, "Slouch: sink into the chair, round your shoulders and upper back."),
    Step("F", "look_down", 30, "Sit upright but look down at your phone or desk."),
    Step("G", "turning", 30, "Sit upright and turn to look beside you, back and forth."),
    Step("H", "lean_in", 30, "Lean toward the screen from the hips, back straight, as if reading small text."),
    Step("I", "recline", 30, "Lean back in your chair, relaxed."),
    Step("J", "out_of_frame", 20, "Stand up or move partly out of the camera view."),
    Step("K", "upright_return", 45, "Sit back down upright and work normally."),
)

CALIBRATION = "A"
SLOUCH_REFERENCE = "A2"
UPRIGHT = ("B", "K")
SLOUCH = ("D", "E")
MILD = ("C",)
CONFOUNDERS = ("F", "G", "H", "I")
ABSENT = ("J",)

NAMES = {s.code: s.name for s in PROTOCOL}
