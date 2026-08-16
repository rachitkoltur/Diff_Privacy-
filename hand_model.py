"""
hand_model.py
=============
Generalized underactuated (whippletree) prosthetic-hand actuator dynamics.

This is NOT a manual hand: a prime-mover actuator pulls a differential
(whippletree) tendon, and per-finger travel is set by actuator/solenoid stops.
We model each finger's mechanical response to a position COMMAND as a critically
/ well-damped 2nd-order servo:

        x'' + 2 zeta wn x' + wn^2 x = wn^2 u(t)

where u(t) is the commanded normalized displacement and x(t) is the physical
finger position. Discretized with a stable state-space (zero-order hold).

The point of the experiment: if the command u(t) is the RAW local estimate the
motion is smooth ("LOCAL COMPUTE"); if u(t) is reconstructed from the NOISY
telemetry ("CLOUD COMPUTE"), the high-frequency Laplace noise excites the
actuator and the finger visibly spasms. This reproduces the draft's "Shaky Arm"
demonstration in simulation, and quantifies it as tracking RMSE.
"""
from __future__ import annotations
import numpy as np
from config import Config


def _servo_step(x, v, u, wn, zeta, dt):
    a = wn * wn * (u - x) - 2 * zeta * wn * v
    v = v + a * dt
    x = x + v * dt
    return x, v


def simulate_finger(command, cfg: Config, saturate=True):
    """Integrate the servo ODE for a commanded trajectory (per control step).

    A real finger cannot travel past its mechanical stops, and a real motor
    driver saturates its command. With saturate=True we clip the command to the
    physical range [0, travel_max] and the finger position to a small overshoot
    band, so a noisy command produces bounded but violent chatter between the
    stops (the realistic 'shaky arm') rather than flying off to infinity."""
    h = cfg.hand
    dt = h.dt_ctrl
    x, v = 0.0, 0.0
    lo, hi = -0.15, h.travel_max + 0.15
    out = np.zeros(len(command))
    for i, u in enumerate(command):
        if saturate:
            u = min(max(float(u), 0.0), h.travel_max)
        x, v = _servo_step(x, v, float(u), h.wn, h.zeta, dt)
        if saturate and (x < lo or x > hi):
            x = min(max(x, lo), hi)
            v = 0.0  # hit a mechanical stop -> velocity killed
        out[i] = x
    return out


def upsample_to_control(sig_5hz, cfg: Config):
    """Hold telemetry-rate (5 Hz) commands to the control rate (50 Hz)."""
    reps = int(round((1.0 / cfg.hand.dt_ctrl) / cfg.signal.telemetry_hz))
    return np.repeat(sig_5hz, reps), reps


def tracking_rmse(reference, actual):
    return float(np.sqrt(np.mean((reference - actual) ** 2)))
