"""
General MIDI instrument registry and default dimension→instrument mapping.
"""

from __future__ import annotations


# Curated subset of General MIDI program numbers (0-indexed)
GENERAL_MIDI_INSTRUMENTS: dict[int, str] = {
    0: "Acoustic Grand Piano",
    1: "Bright Acoustic Piano",
    2: "Electric Grand Piano",
    4: "Electric Piano 1",
    5: "Electric Piano 2",
    6: "Harpsichord",
    8: "Celesta",
    9: "Glockenspiel",
    10: "Music Box",
    11: "Vibraphone",
    12: "Marimba",
    13: "Xylophone",
    14: "Tubular Bells",
    24: "Acoustic Guitar (nylon)",
    25: "Acoustic Guitar (steel)",
    26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)",
    32: "Acoustic Bass",
    33: "Electric Bass (finger)",
    40: "Violin",
    41: "Viola",
    42: "Cello",
    43: "Contrabass",
    46: "Orchestral Harp",
    56: "Trumpet",
    57: "Trombone",
    60: "French Horn",
    65: "Alto Sax",
    66: "Tenor Sax",
    71: "Clarinet",
    73: "Flute",
    74: "Recorder",
    79: "Ocarina",
}

# ── Name → program lookup ────────────────────────────────────

_NAME_TO_PROGRAM: dict[str, int] = {}
for _prog, _name in GENERAL_MIDI_INSTRUMENTS.items():
    _NAME_TO_PROGRAM[_name.lower()] = _prog
    _NAME_TO_PROGRAM[_name.lower().split("(")[0].strip()] = _prog

_ALIASES = {
    "piano": 0, "guitar": 25, "bass": 33, "violin": 40,
    "viola": 41, "cello": 42, "harp": 46, "trumpet": 56,
    "trombone": 57, "horn": 60, "sax": 66, "clarinet": 71,
    "flute": 73, "strings": 40,
}
_NAME_TO_PROGRAM.update(_ALIASES)


def resolve_program(name_or_int: str | int) -> int:
    """Convert an instrument name or program number to a valid GM program."""
    if isinstance(name_or_int, int):
        if 0 <= name_or_int <= 127:
            return name_or_int
        raise ValueError(f"MIDI program must be 0–127, got {name_or_int}")
    key = name_or_int.lower().strip()
    if key in _NAME_TO_PROGRAM:
        return _NAME_TO_PROGRAM[key]
    available = ", ".join(sorted(GENERAL_MIDI_INSTRUMENTS.values()))
    raise ValueError(f"Unknown instrument '{name_or_int}'. Available: {available}")


# Pleasing default instruments for dimensions 0, 1, 2, ...
DEFAULT_DIM_INSTRUMENTS: list[int] = [
    0,   # Acoustic Grand Piano
    42,  # Cello
    73,  # Flute
    40,  # Violin
    25,  # Acoustic Guitar (steel)
    11,  # Vibraphone
    56,  # Trumpet
    71,  # Clarinet
    46,  # Orchestral Harp
    66,  # Tenor Sax
    12,  # Marimba
    60,  # French Horn
    5,   # Electric Piano 2
    41,  # Viola
    43,  # Contrabass
    14,  # Tubular Bells
]
