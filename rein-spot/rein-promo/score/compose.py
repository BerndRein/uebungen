"""Original score + sound design for the Rein Campus spot.

96 BPM, D major, 12 bars x 2.5 s = 30 s. Every hit is placed on the same
timestamps the composition uses (index.html), so picture and sound lock.
Run:  python3 score/compose.py   ->  assets/audio/mix.wav
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
DUR = 30.0
N = int(SR * DUR)
B = 60 / 96          # beat
BAR = 4 * B          # 2.5 s
rng = np.random.default_rng(7)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def buf():
    return np.zeros((2, N))


def add(bus, sig, t, gain=1.0, pan=0.0):
    """Mix mono or stereo `sig` into `bus` at time t (s) with equal-power pan."""
    i = int(round(t * SR))
    if sig.ndim == 1:
        a = (pan + 1) * np.pi / 4
        sig = np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)
    if i < 0:
        sig = sig[:, -i:]
        i = 0
    n = min(sig.shape[1], N - i)
    if n > 0:
        bus[:, i:i + n] += sig[:, :n] * gain


def env_adsr(n, a, d, s, r, sr=SR):
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    sus = max(0, n - a - d - r)
    e = np.concatenate([
        np.linspace(0, 1, max(a, 1)) ** 1.6,
        np.linspace(1, s, max(d, 1)),
        np.full(sus, s),
        np.linspace(s, 0, max(r, 1)) ** 1.3,
    ])
    return e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))


def lowpass(x, fc, order=2):
    b, a = signal.butter(order, min(fc, SR * 0.45) / (SR / 2), "low")
    return signal.lfilter(b, a, x, axis=-1)


def highpass(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), "high")
    return signal.lfilter(b, a, x, axis=-1)


# ---------------------------------------------------------------- instruments
def pad_note(f, dur, cutoff):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for k, det in enumerate([-0.11, -0.05, 0.0, 0.05, 0.11]):
        ff = f * 2 ** (det / 12)
        ph = rng.random()
        saw = 2 * ((t * ff + ph) % 1.0) - 1
        side = (k - 2) / 2.0
        out[0] += saw * (1 - 0.35 * side)
        out[1] += saw * (1 + 0.35 * side)
    out = lowpass(out, cutoff, 2)
    out = lowpass(out, cutoff * 1.6, 1)
    return out * env_adsr(n, 0.55, 0.4, 0.85, 1.1) * 0.09


def piano(f, vel=1.0, dur=2.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(1, 9):
        fk = k * f * np.sqrt(1 + 0.00035 * k * k)
        if fk > SR * 0.45:
            break
        amp = (1 / k ** 1.35) * (0.6 + 0.4 * vel)
        dec = 1.6 + k * 0.9 + (f / 600)
        x += amp * np.sin(2 * np.pi * fk * t + rng.random()) * np.exp(-t * dec)
    hammer = lowpass(rng.standard_normal(n) * np.exp(-t * 180), 2500) * 0.25
    x = (x + hammer) * (1 - np.exp(-t * 900))
    x = lowpass(x, 1800 + 3200 * vel)
    return x * 0.22 * vel


def pluck(f, vel=1.0):
    n = int(1.8 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t + 0.3) * np.exp(-t * 6)
    x *= np.exp(-t * 3.2) * (1 - np.exp(-t * 600))
    return x * 0.16 * vel


def bell(f, dur=6.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    idx = 2.4 * np.exp(-t * 1.4)
    x = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * 3.5 * t))
    x += 0.4 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t * 1.2)
    return x * np.exp(-t * 0.75) * (1 - np.exp(-t * 400)) * 0.12


def sub(f, dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) + 0.18 * np.sin(4 * np.pi * f * t)
    x = np.tanh(x * 1.4) / np.tanh(1.4)
    return x * env_adsr(n, 0.01, 0.2, 0.8, 0.12) * 0.26 * gain


def kick():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 54 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 7.5)
    click = lowpass(rng.standard_normal(n) * np.exp(-t * 400), 4000) * 0.12
    return np.tanh((x + click) * 1.5) * 0.55


def shaker(vel=1.0):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    x = highpass(rng.standard_normal(n), 6500, 2) * np.exp(-t * 55) * (1 - np.exp(-t * 2000))
    return x * 0.06 * vel


def rim():
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    x = signal.lfilter(*signal.butter(2, [1400 / (SR / 2), 4200 / (SR / 2)], "band"), rng.standard_normal(n))
    return x * np.exp(-t * 32) * 0.18


def blip(f):
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 26) * (1 - np.exp(-t * 3000)) * 0.05


def swell(dur, f0=300, f1=5000, rev=True):
    """Filtered-noise swell that crescendos into the end of `dur`."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal((2, n))
    out = np.zeros((2, n))
    seg = 1024
    for s in range(0, n, seg):
        p = (s + seg / 2) / n
        fc = f0 * (f1 / f0) ** p
        b, a = signal.butter(2, [max(60, fc * 0.6) / (SR / 2), min(fc * 1.6, SR * 0.45) / (SR / 2)], "band")
        out[:, s:s + seg] = signal.lfilter(b, a, x[:, s:s + seg], axis=-1)
    e = (t / dur) ** 2.6 if rev else (1 - t / dur) ** 2
    return out * e * 0.22


def load_sfx(name, speed=1.0, reverse=False):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(ROOT / "assets/sfx" / f"{name}.mp3"), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.astype(np.float64)
    if speed != 1.0:
        idx = np.arange(0, x.shape[1] - 1, speed)
        x = np.vstack([np.interp(idx, np.arange(x.shape[1]), ch) for ch in x])
    if reverse:
        x = x[:, ::-1]
    pk = np.max(np.abs(x)) + 1e-9
    x = x / pk
    # trim leading silence so the transient lands exactly on the cue
    on = np.argmax(np.max(np.abs(x), axis=0) > 0.08)
    if not reverse:
        x = x[:, max(0, on - int(0.002 * SR)):]
    return x


def reverb_ir(rt60=2.6, length=3.2):
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((2, n)) * np.exp(-6.9 * t / rt60)
    ir = lowpass(ir, 5200)
    ir[:, : int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
    return ir / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))


# ---------------------------------------------------------------- harmony
#            bass   pad voicing                  pad cutoff
CHORDS = [
    (43, [59, 62, 66, 69], 900),        # 1  Gmaj9      Ideas scatter.
    (45, [62, 64, 69, 71], 1000),       # 2  Asus       (unresolved)
    (47, [62, 66, 69, 73], 1200),       # 3  Bm(add9)   Connect them.
    (43, [62, 66, 69, 71], 1400),       # 4  Gmaj9
    (38, [62, 64, 66, 69, 74], 2600),   # 5  Dadd9      the drop: graph
    (37, [61, 64, 69, 76], 2600),       # 6  A/C#
    (35, [59, 62, 66, 69], 2400),       # 7  Bm7        canvas
    (43, [59, 62, 66, 71], 2200),       # 8  Gmaj7      → yours
    (40, [55, 59, 62, 66], 1700),       # 9  Em9        yours (intimate)
    (45, [62, 64, 67, 69], 2600),       # 10 A7sus4     plugins → tension
    (38, [61, 64, 66, 69, 74], 2000),   # 11 Dmaj9      logo
    (38, [61, 64, 66, 69, 74], 1600),   # 12 hold
]

pad, keys, bass, drums, fx, sfx = buf(), buf(), buf(), buf(), buf(), buf()

# pad: bars 1–10 one chord per bar, bars 11–12 one long chord
PAD_LVL = [0.38, 0.42, 0.5, 0.56, 0.72, 0.72, 0.66, 0.66, 0.62, 0.45]   # dynamic arc per bar
for i, (b_, notes, fc) in enumerate(CHORDS[:10]):
    t0 = i * BAR
    for m in notes:
        add(pad, pad_note(midi(m), BAR + 1.2, fc), t0 - 0.05, gain=PAD_LVL[i])
for m in CHORDS[10][1] + [50]:
    add(pad, pad_note(midi(m), 5.0, 2000), 25.0 - 0.02, gain=0.85)

# scene 1: one pluck per thought fragment (same 8th-note grid as the picture)
frag_notes = [81, 74, 78, 86, 83, 76, 88, 81, 85, 78]
for i, m in enumerate(frag_notes):
    add(keys, pluck(midi(m), 0.8 + 0.2 * (i % 3 == 0)), i * 0.3125 + 0.04, pan=[-0.6, 0.5, 0.3, -0.4, 0.6, -0.2][i % 6])

# felt piano
def arp(bar, pattern, vel=0.8, step=B / 2):
    notes = CHORDS[bar][1]
    for k, idx in enumerate(pattern):
        if idx is None:
            continue
        m = notes[idx % len(notes)] + 12 * (idx // len(notes))
        v = vel * (1.0 if k % 4 == 0 else 0.78)
        add(keys, piano(midi(m), v), bar * BAR + k * step, pan=-0.25 + 0.5 * ((m - 60) / 24))

for bar in (2, 3):                       # Connect them: gentle quarter notes
    arp(bar, [0, 2, 1, 3], 0.7, B)
for bar in range(4, 8):                  # graph + canvas: flowing 8ths
    arp(bar, [0, 2, 4, 2, 1, 3, 5, 3], 0.82)
arp(8, [0, 2, 4, 2, 1, 3, 5, 6], 0.85)            # three fields: lifts
arp(9, [0, 2, 3, None, 1, 3, None, None], 0.6)   # parent quote: thinner
# logo: the theme resolves
for k, m in enumerate([74, 78, 81, 86]):
    add(keys, piano(midi(m), 0.9 - 0.08 * k, 4.5), 25.0 + k * 0.08, pan=-0.2 + 0.14 * k)
add(keys, piano(midi(62), 0.8, 4.5), 25.0)
add(keys, piano(midi(85), 0.55, 4.0), 27.5, pan=0.3)
add(keys, piano(midi(81), 0.5, 4.0), 27.5 + B, pan=-0.2)

# sub bass
for i, (b_, notes, fc) in enumerate(CHORDS[:10]):
    if i < 2:
        continue
    if i in (2, 3, 9):
        add(bass, sub(midi(b_), BAR - 0.05, 0.55), i * BAR)
    else:
        for beat in range(4):
            add(bass, sub(midi(b_), B - 0.03, 0.9), i * BAR + beat * B)
add(bass, sub(midi(38), 4.6, 1.0), 25.0)

# drums: half-time from the drop, out for "yours", four-on-floor for plugins
kick_times, shk_times, rim_times = [], [], []
for bar in range(4, 9):
    for beat in range(4):
        t = bar * BAR + beat * B
        if beat in (0, 2):
            kick_times.append(t)
        if beat in (1, 3):
            rim_times.append(t)
for k in range(int((25.0 - 10.0) / (B / 4))):
    t = 10.0 + k * B / 4
    if t >= 22.5:
        continue
    shk_times.append((t, 1.0 if k % 2 == 0 else 0.55))
for t in kick_times:
    add(drums, kick(), t)
for t in rim_times:
    add(drums, rim(), t, pan=0.15)
for t, v in shk_times:
    add(drums, shaker(v), t, pan=0.35 if int(t / (B / 4)) % 2 else -0.35)

# sidechain-style duck of pad + bass under the kick
duck = np.ones(N)
for t in kick_times:
    i = int(t * SR)
    n = int(0.32 * SR)
    seg = 1 - 0.45 * np.exp(-np.arange(n) / SR * 11)
    duck[i:i + n] = np.minimum(duck[i:i + n], seg[: max(0, min(n, N - i))])
pad = highpass(pad * duck, 140, 2)
bass = lowpass(bass * duck, 220, 2) * 0.42

# plan ticks: a small bright two-note figure each time a box is checked
for k, t in enumerate([5.3, 6.55, 7.8, 9.05, 9.05 + 0.15]):
    add(fx, blip(midi([93, 95, 97, 100, 102][k])), t, gain=1.6, pan=-0.5)
    add(fx, blip(midi([98, 100, 102, 105, 107][k])), t + 0.06, gain=1.0, pan=-0.5)

# swells into the big cuts
add(fx, swell(0.9, 400, 6000), 4.05, gain=0.5)          # notes gathered into the plan
add(fx, swell(0.7, 300, 7000), 9.3, gain=0.75)          # into the hero
add(fx, swell(0.65, 300, 9000), 24.35, gain=0.8)        # into the logo
add(fx, bell(midi(86)), 25.0, gain=1.0, pan=0.1)
add(fx, bell(midi(93)), 25.04, gain=0.45, pan=-0.2)

# ---------------------------------------------------------------- sfx (bundled library)
S = {n: load_sfx(n) for n in ["key-press", "click", "click-soft", "pop", "whoosh-short", "sparkle", "impact-bass-1"]}
S["whoosh-rev"] = load_sfx("whoosh", reverse=True)
for k in range(10):                                             # sticky notes slap on
    add(sfx, load_sfx("pop", 0.92 + 0.05 * (k % 4)), 0.04 + k * 0.3125, 0.12, pan=[-0.6, -0.3, 0.0, 0.3, 0.6][k % 5])
for k, t in enumerate([5.3, 6.55, 7.8, 9.05]):                  # boxes ticked
    add(sfx, S["click-soft"], t, 0.22, pan=-0.5)
add(sfx, S["whoosh-rev"], 10.0 - S["whoosh-rev"].shape[1] / SR, 0.18)
add(sfx, S["impact-bass-1"], 10.0, 0.36)
add(sfx, S["sparkle"], 10.4, 0.12, pan=-0.3)
for k, t in enumerate([15.0, 15.3125, 15.625]):                 # photos land
    add(sfx, load_sfx("whoosh-short", 1.1 + 0.1 * k), t - 0.05, 0.09, pan=0.5)
for k, t in enumerate([16.25, 16.40625, 16.5625]):              # chips
    add(sfx, S["click-soft"], t, 0.12, pan=-0.4 + 0.2 * k)
for k, t in enumerate([19.0, 19.3125, 19.625]):                 # three cards
    add(sfx, load_sfx("pop", 1.0 + 0.06 * k), t, 0.12, pan=[-0.5, 0.0, 0.5][k])
add(sfx, S["whoosh-short"], 22.05, 0.1)
add(sfx, S["impact-bass-1"], 25.0, 0.3)
add(sfx, S["sparkle"], 25.3, 0.12, pan=0.15)
add(sfx, S["click"], 27.5, 0.12)

# ---------------------------------------------------------------- mix
ir = reverb_ir()
music = pad * 1.0 + keys * 1.0 + fx * 1.0
send = pad * 0.5 + keys * 0.7 + fx * 0.8 + sfx * 0.12 + drums * 0.08
wet = np.vstack([signal.fftconvolve(send[c], ir[c])[:N] for c in range(2)]) * 0.55
mix = music + bass * 1.0 + drums * 0.62 + sfx * 1.0 + wet
mix = highpass(mix, 40, 4)
# gentle glue + soft clip (normalised first so the drive is predictable)
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.25) / np.tanh(1.25)
fade = np.ones(N)
fo = int(1.4 * SR)
fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
mix *= fade
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 0.89

out = ROOT / "assets/audio"
out.mkdir(parents=True, exist_ok=True)
pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2").tobytes()
# linear loudness match to -14 LUFS (keeps the dynamic arc), brickwall at -1 dBTP
meas = subprocess.run(
    ["ffmpeg", "-v", "info", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-", "-af", "ebur128", "-f", "null", "-"],
    input=pcm, capture_output=True).stderr.decode()
integ = float(meas.rsplit("I:", 1)[1].split("LUFS")[0])
gain_db = -14.0 - integ
subprocess.run(
    ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-",
     "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.89:attack=3:release=60:level=disabled", "-ar", str(SR), str(out / "mix.wav")],
    input=pcm, check=True)
print(f"integrated before {integ:.1f} LUFS, gain {gain_db:+.1f} dB")
print("wrote", out / "mix.wav")
