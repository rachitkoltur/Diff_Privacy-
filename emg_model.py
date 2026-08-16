"""
emg_model.py
============
Literature-grounded generative model of surface EMG (sEMG) with GROUND-TRUTH
task and keystroke labels, plus the edge-DSP feature extractor (LFA, HFV, RMS).

Why a generative model (not a downloaded dataset)?
--------------------------------------------------
The threat we study is keystroke inference from prosthetic EMG telemetry. No
public dataset pairs amputee prosthetic telemetry with per-keystroke ground
truth, and this environment has no dataset network access. A transparent
generative model is the standard tool for privacy-mechanism evaluation: it lets
us place EXACTLY KNOWN keystroke events into the signal and then measure how
well an adversary can recover them, which is impossible without ground truth.

The model is grounded in the sEMG literature:
  * usable band 20-500 Hz, dominant energy 50-150 Hz (Basmajian & De Luca 1985);
  * gross (macro) contractions -> strong low-band, sustained envelope;
  * fine (micro) finger movements / keystrokes -> brief high-band transients.
These qualitative facts, not any tuned result, drive the band/gain choices in
config.py.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt
from dataclasses import dataclass
from typing import List
from config import Config

REST, GRASP, TYPE = 0, 1, 2
TASK_NAMES = {REST: "rest", GRASP: "grasp", TYPE: "type"}


@dataclass
class Session:
    raw: np.ndarray            # raw sEMG samples (normalized units)
    fs: int
    task_per_sample: np.ndarray  # int task label per raw sample
    keystroke_times: np.ndarray  # sample indices of keystroke onsets
    keystroke_finger: np.ndarray # finger id (0..3) for each keystroke
    win_samples: int
    # per-window (telemetry-rate) arrays, filled by extract_features:
    lfa: np.ndarray = None
    hfv: np.ndarray = None
    rms: np.ndarray = None
    win_task: np.ndarray = None       # majority task label in window
    win_keystroke: np.ndarray = None  # 1 if a keystroke onset falls in window
    win_time: np.ndarray = None


def _bandpass_sos(lo, hi, fs, order=4):
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def _band(x, lo, hi, fs):
    return sosfiltfilt(_bandpass_sos(lo, hi, fs), x)


def make_task_schedule(rng, cfg: Config, duration_s: float) -> List[tuple]:
    """Randomized sequence of (task, seconds) segments."""
    fs = cfg.signal.fs_raw
    segs, t = [], 0.0
    # start resting, then alternate among tasks with random-ish durations
    tasks_cycle = [REST, GRASP, TYPE, REST, TYPE, GRASP, TYPE, REST]
    i = 0
    while t < duration_s:
        task = tasks_cycle[i % len(tasks_cycle)]
        dur = float(rng.uniform(3.0, 7.0))
        segs.append((task, dur))
        t += dur
        i += 1
    return segs


def generate_session(cfg: Config, duration_s: float = 200.0, seed: int | None = None) -> Session:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    s = cfg.signal
    fs = s.fs_raw
    n = int(duration_s * fs)

    # ---- per-sample task labels + time-varying band envelopes ----
    task_per_sample = np.zeros(n, dtype=int)
    g_low = np.zeros(n)
    g_mid = np.zeros(n)
    g_high = np.zeros(n)

    segs = make_task_schedule(rng, cfg, duration_s)
    idx = 0
    keystroke_times, keystroke_finger = [], []
    for task, dur in segs:
        m = int(dur * fs)
        j0, j1 = idx, min(idx + m, n)
        if j0 >= n:
            break
        task_per_sample[j0:j1] = task
        # baseline gains for this task
        g_low[j0:j1] = s.gain_low[task]
        g_mid[j0:j1] = s.gain_mid[task]
        g_high[j0:j1] = s.gain_high[task]
        # smooth onset/offset ramp (muscle recruitment) to avoid step artifacts
        ramp = min(int(0.15 * fs), (j1 - j0) // 2)
        if ramp > 1:
            w = np.linspace(0, 1, ramp)
            for g in (g_low, g_mid, g_high):
                g[j0:j0 + ramp] *= w
                g[j1 - ramp:j1] *= w[::-1]
        # inject keystroke bursts during typing
        if task == TYPE:
            burst = int(s.keystroke_burst_ms / 1000 * fs)
            # Poisson keystroke onsets
            n_keys = rng.poisson(s.keystroke_rate_hz * dur)
            onsets = np.sort(rng.uniform(j0, max(j0 + 1, j1 - burst), size=n_keys).astype(int))
            for on in onsets:
                fin = int(rng.integers(0, cfg.privacy.n_fingers))
                keystroke_times.append(on)
                keystroke_finger.append(fin)
                e0, e1 = on, min(on + burst, n)
                # raised-cosine burst envelope
                bw = np.hanning(max(2, e1 - e0))
                g_high[e0:e1] += s.keystroke_gain_high * bw
                g_low[e0:e1] += s.keystroke_gain_low * bw
        idx = j1

    # ---- synthesize sEMG as sum of band-limited stochastic components ----
    def band_noise(lo, hi):
        return _band(rng.standard_normal(n), lo, hi, fs)

    low = band_noise(*s.low_band)
    mid = band_noise(*s.mid_band)
    high = band_noise(*s.high_band)
    # normalize each band to unit std so gains are interpretable
    for arr in (low, mid, high):
        arr /= (arr.std() + 1e-9)

    raw = g_low * low + g_mid * mid + g_high * high
    raw += s.measurement_noise * rng.standard_normal(n)

    return Session(
        raw=raw, fs=fs, task_per_sample=task_per_sample,
        keystroke_times=np.array(keystroke_times, dtype=int),
        keystroke_finger=np.array(keystroke_finger, dtype=int),
        win_samples=cfg.win_samples,
    )


def extract_features(sess: Session, cfg: Config) -> Session:
    """Edge-DSP feature extraction, once per 200 ms window (telemetry rate).

    LFA_norm : normalized low-band RMS (gross muscle contraction / macro-grasp)
    HFV_norm : normalized high-band variance (micro-dexterity jitter)
    RMS      : normalized broadband RMS telemetry feature (Delta f = 0.1)
    """
    s = cfg.signal
    fs = sess.fs
    W = sess.win_samples
    x = sess.raw
    n_win = len(x) // W

    low = _band(x, *s.low_band, fs)
    high = _band(x, *s.high_band, fs)

    lfa = np.zeros(n_win)
    hfv = np.zeros(n_win)
    rms = np.zeros(n_win)
    win_task = np.zeros(n_win, dtype=int)
    win_key = np.zeros(n_win, dtype=int)
    win_time = np.zeros(n_win)

    key_set = set(sess.keystroke_times.tolist())
    ktimes = sess.keystroke_times
    for i in range(n_win):
        a, b = i * W, (i + 1) * W
        seg = x[a:b]
        rms[i] = np.sqrt(np.mean(seg ** 2))
        lfa[i] = np.sqrt(np.mean(low[a:b] ** 2))          # low-band amplitude
        hfv[i] = np.var(high[a:b])                         # high-band variance
        win_task[i] = np.bincount(sess.task_per_sample[a:b]).argmax()
        win_key[i] = int(np.any((ktimes >= a) & (ktimes < b)))
        win_time[i] = a / fs

    # FIXED normalization by enrollment constants (data-independent), then clip.
    # This keeps the features in [0,1] without using the current session's own
    # statistics, so the risk score and context sensitivity are well-defined.
    sess.lfa = np.clip(lfa / cfg.signal.calib_lfa, 0.0, 1.0)
    sess.hfv = np.clip(hfv / cfg.signal.calib_hfv, 0.0, 1.0)
    sess.rms = np.clip(rms / cfg.signal.calib_rms, 0.0, 1.0)
    sess.win_task = win_task
    sess.win_keystroke = win_key
    sess.win_time = win_time
    return sess


def generate_tendon_vector(sess: Session, cfg: Config, seed: int | None = None) -> np.ndarray:
    """Per-timestep 4-D tendon displacement vector D(t) in [0,1]^4.

    Generalized underactuated/whippletree hand (actuator/solenoid-set stops):
      * rest  -> fingers extended (d ~ 0)
      * grasp -> all fingers retract together via the whippletree (d ~ 0.8)
      * type  -> individual finger makes a brief retraction at each keystroke
    """
    rng = np.random.default_rng((cfg.seed + 101) if seed is None else seed)
    n_win = len(sess.win_task)
    F = cfg.privacy.n_fingers
    D = np.zeros((n_win, F))
    base = np.zeros(F)
    for i in range(n_win):
        task = sess.win_task[i]
        if task == GRASP:
            target = np.full(F, 0.80)
        elif task == REST:
            target = np.zeros(F)
        else:  # TYPE: small resting flexion, keystroke handled below
            target = np.full(F, 0.15)
        base = 0.6 * base + 0.4 * target  # 1st-order smoothing (tendon dynamics)
        D[i] = base
    # add keystroke retractions on the specific finger
    W = sess.win_samples
    for on, fin in zip(sess.keystroke_times, sess.keystroke_finger):
        wi = on // W
        if wi < n_win:
            D[wi, fin] = min(1.0, D[wi, fin] + 0.6)
    D += 0.01 * rng.standard_normal(D.shape)  # tiny mechanical noise
    return np.clip(D, 0.0, 1.0)
