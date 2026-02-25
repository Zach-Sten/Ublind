"""ublind.hlp — Helper lookups for scales, instruments, and usage."""

from __future__ import annotations


def instruments() -> None:
    """Print available instruments and their aliases."""
    from ublind._core.instruments import GENERAL_MIDI_INSTRUMENTS, _ALIASES

    print("╔══════════════════════════════════════════════╗")
    print("║         ublind — Available Instruments       ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Use with dim_inst_map={0: 'name', ...}     ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Show aliases (the friendly names)
    alias_to_gm = {}
    for alias, prog in sorted(_ALIASES.items()):
        gm_name = GENERAL_MIDI_INSTRUMENTS.get(prog, f"Program {prog}")
        alias_to_gm[alias] = (prog, gm_name)

    for alias, (prog, gm_name) in sorted(alias_to_gm.items()):
        print(f"  {alias:<14s}  →  {gm_name} (GM {prog})")

    print()
    print("  Or pass any GM program number directly: {0: 42}")
    print("  Full GM list: ub.hlp.gm_table()")


def scales() -> None:
    """Print available scales."""
    from ublind._core.notes import SCALES

    print("╔══════════════════════════════════════════════╗")
    print("║           ublind — Available Scales          ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Use with scale='name' in preprocess()      ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    vibes = {
        "pentatonic": "always sounds good, safe default",
        "major": "happy, bright",
        "minor": "sad, dark, dramatic",
        "blues": "jazzy, soulful",
        "dorian": "jazzy minor, sophisticated",
        "mixolydian": "bluesy major, laid back",
        "whole_tone": "dreamy, floating, impressionist",
        "chromatic": "all 12 semitones, atonal chaos",
    }

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for name, intervals in SCALES.items():
        notes = [NOTE_NAMES[i % 12] for i in intervals]
        vibe = vibes.get(name, "")
        print(f"  {name:<14s}  {' '.join(notes):<20s}  {vibe}")

    print()
    print("  Transpose with root=N  (0=C, 2=D, 4=E, 5=F, 7=G, 9=A, 11=B)")


def gm_table() -> None:
    """Print the full General MIDI instrument table."""
    from ublind._core.instruments import GENERAL_MIDI_INSTRUMENTS

    print("╔══════════════════════════════════════════════╗")
    print("║       General MIDI Instrument Table          ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Use with dim_inst_map={0: number, ...}     ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    for prog in sorted(GENERAL_MIDI_INSTRUMENTS.keys()):
        name = GENERAL_MIDI_INSTRUMENTS[prog]
        print(f"  {prog:>3d}  {name}")


def quickstart() -> None:
    """Print a quick usage guide."""
    print("""
╔══════════════════════════════════════════════╗
║            ublind — Quick Start              ║
╚══════════════════════════════════════════════╝

  import ublind as ub

  ── Sweep mode (every dim = a voice) ──────────

  ub.pp.preprocess(adata, embedding="X_umap",
      time=10, scale="pentatonic",
      dim_inst_map={0: "piano", 1: "cello"})
  ub.tl.render(adata, "sweep.wav")
  ub.pl.sweep(adata, color_by="cell_type")

  ── Cluster mode (one cluster at a time) ──────

  ub.pp.preprocess_clusters(adata,
      cluster_key="cell_type", time=10,
      order="largest", scale="pentatonic")
  ub.tl.render(adata, "clusters.wav")
  ub.pl.sweep_clusters(adata)

  ── Spatial sweep (across tissue) ─────────────

  ub.pp.preprocess_clusters(adata, ...)   # first
  ub.pl.sweep_spatial(adata,
      spatial_key="spatial", sweep_axis="both",
      color_by="cell_type")

  ── Interactive (click to hear) ───────────────

  ub.pl.interactive(adata, color_by="cell_type")

  ── Helpers ────────────────────────────────────

  ub.hlp.scales()       # available scales
  ub.hlp.instruments()  # available instruments
  ub.hlp.gm_table()     # full GM instrument list
""")
