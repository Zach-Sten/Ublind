"""
Audio renderers.

  MidiRenderer       — .mid file via pure Python (no deps)
  SynthRenderer      — .wav via numpy additive synthesis
  FluidSynthRenderer — .wav via FluidSynth + SoundFont
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from ublind._core.notes import NoteEvent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MIDI renderer (zero dependencies)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MidiRenderer:
    """Write a standard multi-track MIDI file."""

    def __init__(self, ticks_per_beat: int = 480):
        self.tpb = ticks_per_beat

    def render(
        self,
        events: Sequence[NoteEvent],
        output: Path,
        *,
        tempo_bpm: float = 120.0,
    ) -> None:
        output = Path(output)
        tpb = self.tpb
        us_per_beat = int(60_000_000 / tempo_bpm)
        sec_per_tick = us_per_beat / 1_000_000 / tpb

        events = _normalize_time(events)

        # Group channels for program changes
        channels_used: dict[int, int] = {}
        for ev in events:
            channels_used.setdefault(ev.channel, ev.instrument)

        track_events: list[tuple[int, bytes]] = []

        # Tempo meta-event at tick 0
        tempo_bytes = us_per_beat.to_bytes(3, "big")
        track_events.append((0, b"\xff\x51\x03" + tempo_bytes))

        # Program changes
        for ch, prog in channels_used.items():
            track_events.append((0, bytes([0xC0 | (ch & 0x0F), prog & 0x7F])))

        for ev in events:
            tick_on = int(ev.time / sec_per_tick)
            tick_off = tick_on + max(1, int(ev.duration / sec_per_tick))
            ch = ev.channel & 0x0F
            note = max(0, min(127, ev.pitch))
            vel = max(0, min(127, ev.velocity))
            track_events.append((tick_on, bytes([0x90 | ch, note, vel])))
            track_events.append((tick_off, bytes([0x80 | ch, note, 0])))

        track_events.sort(
            key=lambda x: (x[0], 0 if x[1][0] & 0xF0 == 0x80 else 1)
        )

        raw = bytearray()
        prev = 0
        for tick, data in track_events:
            raw.extend(_vlq(tick - prev))
            raw.extend(data)
            prev = tick
        raw.extend(_vlq(0))
        raw.extend(b"\xff\x2f\x00")

        track_chunk = b"MTrk" + len(raw).to_bytes(4, "big") + bytes(raw)
        header = (
            b"MThd"
            + (6).to_bytes(4, "big")
            + struct.pack(">HHH", 0, 1, tpb)
        )
        output.write_bytes(header + track_chunk)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Additive-synthesis renderer (numpy only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SynthRenderer:
    """
    Render to WAV using additive synthesis with per-instrument timbres,
    ADSR envelopes, and multi-tap reverb.

    Parameters
    ----------
    sample_rate : int
    gain : float
        Master gain (0–1).
    reverb : float
        Reverb wet mix (0–1).
    attack, decay, sustain_level, release : float
        ADSR envelope parameters.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        gain: float = 0.35,
        attack: float = 0.01,
        decay: float = 0.08,
        sustain_level: float = 0.55,
        release: float = 0.20,
        reverb: float = 0.25,
    ):
        self.sr = sample_rate
        self.gain = gain
        self.attack = attack
        self.decay = decay
        self.sustain_level = sustain_level
        self.release = release
        self.reverb = reverb

    def render(
        self,
        events: Sequence[NoteEvent],
        output: Path,
        *,
        tempo_bpm: float = 120.0,
    ) -> None:
        output = Path(output)
        sr = self.sr

        if not events:
            _write_wav(output, np.zeros(sr, dtype=np.float32), sr)
            return

        events = _normalize_time(events)
        max_release = max(_get_timbre(e.instrument)["adsr"][3] for e in events)
        end = max(e.time + e.duration + max_release for e in events) + 0.5
        n = int(end * sr)
        buf = np.zeros(n, dtype=np.float64)

        for ev in events:
            timbre = _get_timbre(ev.instrument)
            freq = _midi_to_freq(ev.pitch)
            vel = ev.velocity / 127.0
            note = self._synth_note(freq, ev.duration, vel, timbre)
            start = int(ev.time * sr)
            end_i = start + len(note)
            if end_i > n:
                note = note[: n - start]
                end_i = n
            if start < n:
                buf[start:end_i] += note

        if self.reverb > 0:
            buf = _apply_reverb(buf, sr, self.reverb)

        peak = np.max(np.abs(buf))
        if peak > 0:
            buf = buf / peak * self.gain
        buf = np.tanh(buf * 1.5) / np.tanh(1.5)

        _write_wav(output, buf.astype(np.float32), sr)

    def render_bytes(
        self,
        events: Sequence[NoteEvent],
        total_time: float,
        tempo_bpm: float = 120.0,
    ) -> bytes:
        """Render to WAV and return bytes (for spatial sweep re-rendering)."""
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        self.render(events, tmp, tempo_bpm=tempo_bpm)
        data = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return data

    def _synth_note(
        self,
        freq: float,
        duration: float,
        vel: float,
        timbre: dict,
    ) -> np.ndarray:
        sr = self.sr
        adsr = timbre["adsr"]  # (attack, decay, sustain, release)
        release = adsr[3]
        total = duration + release
        n = int(total * sr)
        t = np.arange(n, dtype=np.float64) / sr

        # Vibrato modulation
        vib_hz = timbre.get("vibrato_hz", 0)
        vib_depth = timbre.get("vibrato_depth", 0)
        if vib_hz > 0 and vib_depth > 0:
            # Vibrato ramps in over the first 0.2s
            vib_ramp = np.minimum(t / 0.2, 1.0)
            vibrato = 1.0 + vib_depth * vib_ramp * np.sin(2 * np.pi * vib_hz * t)
        else:
            vibrato = 1.0

        # Additive harmonics
        wave = np.zeros_like(t)
        for amp, mult, detune, decay_mult in timbre["harmonics"]:
            f = freq * mult * vibrato + detune
            env = np.exp(-(1.0 + decay_mult) * t)
            wave += amp * env * np.sin(2 * np.pi * f * t)

        # Breath / bow noise
        noise_amt = timbre.get("noise_amount", 0)
        if noise_amt > 0:
            rng = np.random.default_rng(int(freq * 1000) & 0xFFFFFFFF)
            noise = rng.normal(0, noise_amt, n)
            wave += noise

        # Per-instrument ADSR envelope
        env = self._adsr_custom(n, duration, *adsr)

        return wave * env * vel

    def _adsr_custom(
        self, n: int, duration: float,
        attack: float, decay: float, sustain_level: float, release: float,
    ) -> np.ndarray:
        """Build an ADSR envelope with per-instrument parameters."""
        sr = self.sr
        a = int(attack * sr)
        d = int(decay * sr)
        r = int(release * sr)
        s_len = max(0, n - a - d - r)

        parts = []
        if a > 0:
            parts.append(np.linspace(0, 1, a))
        if d > 0:
            parts.append(np.linspace(1, sustain_level, d))
        if s_len > 0:
            parts.append(np.full(s_len, sustain_level))
        if r > 0:
            parts.append(np.linspace(sustain_level, 0, r))

        env = np.concatenate(parts) if parts else np.zeros(1)
        if len(env) >= n:
            return env[:n]
        return np.pad(env, (0, n - len(env)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FluidSynth renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FluidSynthRenderer:
    """
    Render via FluidSynth + SoundFont for realistic instruments.

    Requires ``pip install pyfluidsynth`` and a ``.sf2`` file.
    """

    def __init__(
        self,
        soundfont: str | Path | None = None,
        sample_rate: int = 44100,
        gain: float = 0.5,
    ):
        self.soundfont = Path(soundfont) if soundfont else None
        self.sr = sample_rate
        self.gain = gain

    def render(
        self,
        events: Sequence[NoteEvent],
        output: Path,
        *,
        tempo_bpm: float = 120.0,
    ) -> None:
        import fluidsynth

        output = Path(output)
        sf = self._resolve_soundfont()
        fs = fluidsynth.Synth(samplerate=float(self.sr), gain=self.gain)
        sfid = fs.sfload(str(sf))

        events = _normalize_time(events)

        for ev in events:
            fs.program_select(ev.channel, sfid, 0, ev.instrument)

        if not events:
            _write_wav(output, np.zeros(self.sr, dtype=np.float32), self.sr)
            fs.delete()
            return

        end = max(e.time + e.duration for e in events) + 1.0
        n = int(end * self.sr)

        timeline: list[tuple[float, str, NoteEvent]] = []
        for ev in events:
            timeline.append((ev.time, "on", ev))
            timeline.append((ev.time + ev.duration, "off", ev))
        timeline.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))

        buf = np.zeros(n, dtype=np.float32)
        pos = 0
        tidx = 0
        chunk = 512

        while pos < n:
            t = pos / self.sr
            while tidx < len(timeline) and timeline[tidx][0] <= t:
                _, etype, ev = timeline[tidx]
                if etype == "on":
                    fs.noteon(ev.channel, ev.pitch, ev.velocity)
                else:
                    fs.noteoff(ev.channel, ev.pitch)
                tidx += 1
            sz = min(chunk, n - pos)
            samples = fs.get_samples(sz)
            stereo = (
                np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
            )
            if len(stereo) >= sz * 2:
                mono = (stereo[::2] + stereo[1::2]) * 0.5
            else:
                mono = np.zeros(sz, np.float32)
            buf[pos : pos + len(mono)] = mono[: min(len(mono), n - pos)]
            pos += sz

        fs.delete()
        _write_wav(output, buf, self.sr)

    def _resolve_soundfont(self) -> Path:
        if self.soundfont and Path(self.soundfont).expanduser().exists():
            return Path(self.soundfont).expanduser()

        # Check relative to the ublind package location
        import ublind
        pkg_dir = Path(ublind.__file__).resolve().parent
        # Walk up to find project root (where pyproject.toml lives)
        for parent in [pkg_dir] + list(pkg_dir.parents):
            sf = parent / "soundfonts" / "FluidR3_GM.sf2"
            if sf.exists():
                return sf
            if (parent / "pyproject.toml").exists():
                break

        # Common system locations
        for c in [
            Path.home() / "FluidR3_GM.sf2",
            Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
            Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
            Path("/usr/share/sounds/sf2/default-GM.sf2"),
            Path("/opt/homebrew/share/fluidsynth/FluidR3_GM.sf2"),
            Path("/usr/local/share/fluidsynth/FluidR3_GM.sf2"),
            Path.home() / ".local/share/soundfonts/FluidR3_GM.sf2",
        ]:
            if c.exists():
                return c

        raise FileNotFoundError(
            "No SoundFont found. Place FluidR3_GM.sf2 in your project's\n"
            "soundfonts/ folder, or pass soundfont='/path/to/your.sf2'.\n"
            "Download from: https://member.keymusician.com/Member/FluidR3_GM/"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Timbre definitions — each instrument has harmonics AND its own ADSR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# harmonics: (relative_amplitude, harmonic_number, detune_hz, decay_rate_multiplier)
# adsr: (attack_s, decay_s, sustain_level, release_s)

_TIMBRES: dict[str, dict] = {
    "piano": {
        "harmonics": [
            (1.0, 1, 0.0, 1.0), (0.5, 2, 0.3, 2.0), (0.25, 3, 0.5, 3.0),
            (0.12, 4, 0.7, 4.5), (0.06, 5, 0.8, 6.0), (0.03, 6, 1.0, 8.0),
        ],
        "adsr": (0.005, 0.1, 0.3, 0.3),  # sharp attack, fast decay
    },
    "cello": {
        "harmonics": [
            (1.0, 1, 0.0, 0.1), (0.7, 2, 0.5, 0.15), (0.4, 3, 1.0, 0.2),
            (0.2, 4, 1.2, 0.3), (0.1, 5, 1.5, 0.5),
        ],
        "adsr": (0.08, 0.15, 0.8, 0.25),  # slow attack, long sustain
        "vibrato_hz": 5.0, "vibrato_depth": 0.003,
    },
    "flute": {
        "harmonics": [
            (1.0, 1, 0.0, 0.05), (0.08, 2, 0.2, 0.1), (0.02, 3, 0.3, 0.2),
        ],
        "adsr": (0.05, 0.05, 0.85, 0.15),  # breathy attack
        "noise_amount": 0.03,  # breath noise
        "vibrato_hz": 5.5, "vibrato_depth": 0.004,
    },
    "violin": {
        "harmonics": [
            (1.0, 1, 0.0, 0.1), (0.85, 2, 0.8, 0.12), (0.6, 3, 1.2, 0.18),
            (0.35, 4, 1.5, 0.3), (0.2, 5, 1.8, 0.5), (0.1, 6, 2.0, 0.8),
        ],
        "adsr": (0.04, 0.1, 0.75, 0.2),
        "vibrato_hz": 5.5, "vibrato_depth": 0.004,
    },
    "guitar": {
        "harmonics": [
            (1.0, 1, 0.0, 2.5), (0.7, 2, 0.5, 3.5), (0.45, 3, 0.8, 5.0),
            (0.2, 4, 1.0, 7.0), (0.08, 5, 1.2, 10.0),
        ],
        "adsr": (0.002, 0.05, 0.15, 0.5),  # pluck: instant attack, fast decay
    },
    "vibraphone": {
        "harmonics": [
            (1.0, 1, 0.0, 0.3), (0.02, 2, 0.0, 0.5), (0.25, 3, 0.0, 0.6),
            (0.08, 4, 0.0, 1.0),
        ],
        "adsr": (0.001, 0.3, 0.4, 0.8),  # bell-like: instant, long ring
        "vibrato_hz": 6.0, "vibrato_depth": 0.006,
    },
    "trumpet": {
        "harmonics": [
            (1.0, 1, 0.0, 0.1), (0.95, 2, 0.3, 0.12), (0.8, 3, 0.5, 0.15),
            (0.6, 4, 0.7, 0.2), (0.35, 5, 0.8, 0.3), (0.15, 6, 1.0, 0.5),
        ],
        "adsr": (0.03, 0.08, 0.7, 0.1),  # brass attack
        "vibrato_hz": 4.5, "vibrato_depth": 0.002,
    },
    "clarinet": {
        "harmonics": [
            (1.0, 1, 0.0, 0.08), (0.03, 2, 0.2, 0.12),  # weak even harmonics
            (0.65, 3, 0.4, 0.1), (0.02, 4, 0.5, 0.15),   # strong odd harmonics
            (0.35, 5, 0.6, 0.15), (0.01, 6, 0.7, 0.2),
            (0.15, 7, 0.8, 0.2),
        ],
        "adsr": (0.04, 0.06, 0.8, 0.12),
        "vibrato_hz": 4.0, "vibrato_depth": 0.002,
    },
    "harp": {
        "harmonics": [
            (1.0, 1, 0.0, 1.8), (0.35, 2, 0.3, 3.0), (0.12, 3, 0.5, 5.0),
            (0.04, 4, 0.6, 8.0),
        ],
        "adsr": (0.001, 0.15, 0.1, 0.8),  # pluck with long ring
    },
    "sax": {
        "harmonics": [
            (1.0, 1, 0.0, 0.1), (0.75, 2, 0.8, 0.12), (0.65, 3, 1.2, 0.15),
            (0.45, 4, 1.5, 0.2), (0.25, 5, 1.8, 0.3),
        ],
        "adsr": (0.03, 0.08, 0.75, 0.1),
        "vibrato_hz": 5.0, "vibrato_depth": 0.005,
        "noise_amount": 0.015,  # breath
    },
}


def _get_timbre(program: int) -> dict:
    """Map a GM program number to a timbre definition dict."""
    if program <= 7:
        return _TIMBRES["piano"]
    elif program <= 15:
        return _TIMBRES["vibraphone"]
    elif program <= 39:
        return _TIMBRES["guitar"]
    elif 40 <= program <= 41:
        return _TIMBRES["violin"]
    elif 42 <= program <= 43:
        return _TIMBRES["cello"]
    elif 44 <= program <= 47:
        return _TIMBRES["harp"]
    elif 56 <= program <= 63:
        return _TIMBRES["trumpet"]
    elif 64 <= program <= 67:
        return _TIMBRES["sax"]
    elif 71 <= program <= 72:
        return _TIMBRES["clarinet"]
    elif 73 <= program <= 79:
        return _TIMBRES["flute"]
    else:
        return _TIMBRES["piano"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize_time(events: Sequence[NoteEvent]) -> list[NoteEvent]:
    """Shift all events so the earliest starts at t=0."""
    if not events:
        return []
    min_t = min(e.time for e in events)
    if min_t >= 0:
        return list(events)
    return [
        NoteEvent(e.time - min_t, e.duration, e.pitch, e.velocity, e.instrument, e.channel)
        for e in events
    ]


def _vlq(value: int) -> bytes:
    """Encode an integer as a MIDI variable-length quantity."""
    if value < 0:
        value = 0
    buf = [value & 0x7F]
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buf))


def _midi_to_freq(note: int) -> float:
    """MIDI note → frequency in Hz (A4=440)."""
    return 440.0 * 2 ** ((note - 69) / 12.0)


def _apply_reverb(buf: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Simple multi-tap delay reverb."""
    for delay_ms, g in [(23, 0.4), (47, 0.25), (71, 0.15), (113, 0.08)]:
        d = int(delay_ms * sr / 1000)
        delayed = np.zeros_like(buf)
        delayed[d:] = buf[:-d] * g * amount
        buf = buf + delayed
    return buf


def _write_wav(path: Path, buf: np.ndarray, sr: int) -> None:
    """Write a float32 numpy buffer to a 16-bit WAV file."""
    buf = np.clip(buf, -1.0, 1.0)
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
