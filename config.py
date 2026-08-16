"""
config.py
=========
Pre-registered constants for the Context-Aware Local Differential Privacy
(CA-LDP) prosthetic-telemetry simulation.

Every value here is fixed BEFORE any experiment is run (pre-registration), to
avoid the p-hacking / researcher-degrees-of-freedom problem noted in the draft.
Nothing in this file is tuned to make a result look good; the physiological
values are grounded in the surface-EMG literature (Basmajian & De Luca, 1985:
usable sEMG band 20-500 Hz, dominant energy 50-150 Hz).
"""

from dataclasses import dataclass, field
from typing import Tuple

GLOBAL_SEED = 20260722  # fixed master seed for full reproducibility


@dataclass(frozen=True)
class SignalConfig:
    # --- Sampling ---
    fs_raw: int = 500          # raw sEMG sample rate [Hz]  (Nyquist 250 Hz)
    window_ms: int = 200       # DSP feature/telemetry window [ms]
    # samples per window = fs_raw * window_ms/1000 = 100  -> matches RMS N=100
    telemetry_hz: float = 5.0  # broadcast rate = 1 window / 200 ms

    # --- sEMG frequency bands [Hz] (De Luca band; split into low/mid/high) ---
    low_band: Tuple[float, float] = (20.0, 80.0)    # gross contraction (LFA)
    mid_band: Tuple[float, float] = (80.0, 150.0)   # dominant energy band
    high_band: Tuple[float, float] = (150.0, 245.0) # micro-dexterity jitter (HFV)

    # --- Task-dependent band gains (dimensionless activation envelopes) ---
    # (rest, grasp, type) baseline gains for each band.
    gain_low:  Tuple[float, float, float] = (0.10, 1.00, 0.30)
    gain_mid:  Tuple[float, float, float] = (0.08, 0.70, 0.35)
    gain_high: Tuple[float, float, float] = (0.05, 0.12, 0.35)

    # keystroke burst: transient added during typing at each key event
    keystroke_burst_ms: float = 45.0     # duration of a keystroke EMG burst
    keystroke_gain_high: float = 0.85    # extra high-band gain during a burst
    keystroke_gain_low: float = 0.25     # extra low-band gain during a burst
    keystroke_rate_hz: float = 6.0       # mean keystrokes/s while typing (fast typist)

    measurement_noise: float = 0.02      # sensor/ADC additive noise (std, normalized)

    # FIXED feature-normalization constants, determined once at ENROLLMENT and
    # frozen (data-independent at deployment). Features are normalized by these
    # and clipped to [0,1], so the risk score and context sensitivity do not
    # depend on the session's own data.
    calib_lfa: float = 1.1745
    calib_hfv: float = 0.6545
    calib_rms: float = 1.3992


@dataclass(frozen=True)
class PrivacyConfig:
    # --- Pre-registered risk-score / budget parameters (Step 1-2 of draft) ---
    alpha: float = 0.7     # weight on High-Frequency Variance (micro-dexterity)
    beta: float = 0.3      # weight on Low-Frequency Amplitude (macro-grasp)
    k: float = 2.0         # exponential decay constant (calibration value)

    eps_max: float = 8.0   # maximum (loosest) per-timestamp budget (rest/grasp; non-sensitive)
    eps_min: float = 0.05  # REFINEMENT: floor to prevent unbounded noise (b -> inf)

    # --- Neighboring relation and sensitivities ---
    # UNIT OF PRIVACY: a single keystroke EVENT (presence/absence of one keypress).
    # A keystroke occupies at most keystroke_samples_max raw samples of one RMS
    # window, so its L1-sensitivity for the RMS feature is sqrt(m_max / N), NOT the
    # single-sample bound 1/sqrt(N). The single-sample value is kept only for the
    # sample-level relation shown in the sensitivity experiment.
    delta_f_sample: float = 0.1    # sample-level L1-sensitivity of RMS = 1/sqrt(100)
    keystroke_samples_max: int = 30  # pre-registered max raw-sample footprint of ONE keystroke (60 ms @ 500 Hz)
    keystroke_max_concurrent: int = 2  # rollover/held keys: up to K keystrokes may overlap in one window
    delta_d: float = 1.0       # per-finger tendon sensitivity; one keystroke moves <=1 channel by <=1
    n_fingers: int = 4         # tendon vector dimension D = [d1..d4]
    n_rms_samples: int = 100   # N for RMS telemetry feature

    @property
    def delta_f_event(self) -> float:
        """Event-level L1-sensitivity of the windowed RMS, allowing up to K
        overlapping keystrokes (footprint K*m samples): sqrt(K*m/N)."""
        m = self.keystroke_max_concurrent * self.keystroke_samples_max
        return (m / self.n_rms_samples) ** 0.5  # sqrt(60/100)=0.7746

    # --- Sound (output-measurable) context release ---
    # The adaptive budget is set from a coarse activity context that is itself
    # released under DP, so eps(t) is a function of PUBLIC output (post-processing).
    eps_ctx: float = 0.25      # budget spent per CONTEXT RELEASE (not per step)
    ctx_smooth: int = 15       # trailing windows used to smooth the risk (3 s)
    ctx_release_period: int = 15  # release the context once per this many steps (hold between);
    #                               a slowly-varying context need not be re-released at 5 Hz

    @property
    def delta_ctx(self) -> float:
        """Sensitivity of the smoothed risk to one keystroke. R = alpha*HFV - beta*LFA
        depends on BOTH features, and a single keystroke may move both HFV and LFA
        (each clipped to [0,1]); by the triangle inequality |dR| <= alpha*|dHFV| +
        beta*|dLFA| <= alpha + beta. One keystroke spans up to F_tel =
        keystroke_max_concurrent smoothing windows (window overlap), so averaged over
        ctx_smooth windows the smoothed risk changes by at most
        F_tel * (alpha + beta) / ctx_smooth = 2 (alpha + beta) / W = 0.1333."""
        return self.keystroke_max_concurrent * (self.alpha + self.beta) / self.ctx_smooth

    # --- Per-timestamp budget split (REFINEMENT: reconcile EMG + tendon budgets) ---
    # Draft treated the EMG feature (Step 4) and tendon vector (Step 5) with
    # independent budgets. We instead SEQUENTIALLY COMPOSE them so the whole
    # timestamp release is eps(t)-LDP: half the budget to the EMG scalar, half to
    # the 4-D tendon vector (each finger then gets eps(t)/8).
    frac_emg: float = 0.5
    frac_tendon_vector: float = 0.5

    # --- w-event streaming DP (Step 6) ---
    w_window: int = 300        # sliding window (300 * 200 ms = 60 s)
    eps_W: float = 300.0       # total budget over any w-window (Kellaris cap); ~1.0/step avg

    # normalization calibration percentile (features mapped to [0,1] robustly)
    calib_percentile: float = 99.0


@dataclass(frozen=True)
class HandConfig:
    """Generalized underactuated (whippletree) hand with actuator/solenoid stops.
    NOT tied to a manual hand: each finger's travel limit d_i is set by an
    actuator/solenoid, and a shared prime mover pulls the whippletree tendon.
    A simple 2nd-order actuator model turns a position *command* into motion,
    so that a noisy command produces physically-plausible oscillation."""
    wn: float = 12.0      # actuator natural frequency [rad/s]
    zeta: float = 0.7     # damping ratio (well-damped servo)
    dt_ctrl: float = 0.02 # control loop timestep [s] (50 Hz)
    travel_max: float = 1.0


@dataclass(frozen=True)
class Config:
    signal: SignalConfig = field(default_factory=SignalConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    seed: int = GLOBAL_SEED

    @property
    def win_samples(self) -> int:
        return int(self.signal.fs_raw * self.signal.window_ms / 1000)


CONFIG = Config()
