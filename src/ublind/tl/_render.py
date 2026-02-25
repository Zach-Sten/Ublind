"""Render preprocessed scores to audio files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ublind._core.renderers import MidiRenderer, SynthRenderer, FluidSynthRenderer


def render(
    adata,
    output: str = "ublind_output.wav",
    *,
    renderer: Optional[str] = None,
    soundfont: Optional[str] = None,
    gain: float = 0.35,
    reverb: float = 0.25,
    store: bool = True,
) -> Path:
    """
    Render the preprocessed score to an audio file.

    Also stores the WAV bytes in ``adata.uns['ublind']['wav']``
    so the audio persists with the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Must have been preprocessed with ``ub.pp.preprocess()``.
    output : str
        Output path. Extension determines format (``.wav``, ``.mid``).
    renderer : str, optional
        ``"synth"`` (default for .wav), ``"midi"`` (default for .mid),
        or ``"fluidsynth"`` (requires pyfluidsynth + .sf2).
    soundfont : str, optional
        Path to .sf2 SoundFont (for FluidSynth renderer).
    gain : float
        Master volume (0–1).
    reverb : float
        Reverb mix (0–1).
    store : bool
        If True, store WAV bytes in ``adata.uns['ublind']['wav']``.

    Returns
    -------
    Path to the output file.
    """
    ub = _get_uns(adata)
    events = ub["events"]
    tempo = ub["tempo_bpm"]
    output = Path(output)

    if renderer is None:
        renderer = "midi" if output.suffix.lower() == ".mid" else "synth"

    if renderer == "midi":
        r = MidiRenderer()
    elif renderer == "fluidsynth":
        r = FluidSynthRenderer(soundfont=soundfont, gain=gain)
    else:
        r = SynthRenderer(gain=gain, reverb=reverb)

    r.render(events, output, tempo_bpm=tempo)

    # Store WAV bytes in adata.uns so it persists with the object
    if store and output.suffix.lower() == ".wav":
        ub["wav"] = output.read_bytes()

    print(f"ublind: rendered → {output}")
    return output


def to_midi(adata, output: str = "ublind_output.mid") -> Path:
    """Shortcut: render to MIDI."""
    return render(adata, output, renderer="midi")


def to_wav(adata, output: str = "ublind_output.wav", **kwargs) -> Path:
    """Shortcut: render to WAV with built-in synth."""
    return render(adata, output, renderer="synth", **kwargs)


def get_wav_bytes(adata) -> bytes:
    """
    Retrieve stored WAV bytes from adata.uns.

    Can be written back to disk later::

        wav = ub.tl.get_wav_bytes(adata)
        with open("restored.wav", "wb") as f:
            f.write(wav)
    """
    ub = _get_uns(adata)
    if "wav" not in ub:
        raise RuntimeError(
            "No WAV data stored. Run ub.tl.render(adata, 'output.wav') first."
        )
    return ub["wav"]


def play(adata):
    """Play the stored audio inline in a Jupyter notebook."""
    from IPython.display import Audio, display

    wav = get_wav_bytes(adata)
    display(Audio(data=wav, autoplay=False))


def _get_uns(adata) -> dict:
    if "ublind" not in adata.uns:
        raise RuntimeError("No ublind data found. Run ub.pp.preprocess(adata) first.")
    return adata.uns["ublind"]
