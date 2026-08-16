# Context-Aware Local Differential Privacy (CA-LDP) for Prosthetic-Limb Telemetry

A defense against an EMG keystroke side-channel: individual keystrokes are hidden
from prosthetic-limb telemetry while the manufacturer's legitimate aggregate
analytics stay usable. Everything here is a simulation study and is deterministic
under master seed 20260722, so anyone who runs it gets the same numbers.

## What this shows
- The keystroke side channel is real (attacker reads keystrokes near-perfectly
  with no privacy).
- A naive *endogenous* adaptive budget cannot work under local DP (negative
  result: privately releasing an EMG-derived context leaves it at chance).
- A *fixed* budget plus an *exogenous* grip-mode adaptive scheme both drive the
  attacker to chance while keeping legitimate utility, and the exogenous scheme
  beats any single fixed budget.
- The noise sampler is finite-precision safe (discrete Laplace), closing the
  floating-point and timing side channels.

## Run order
    python experiments.py           # E1-E9, writes results.json + figures/
    python realdata_experiment.py   # real 29-participant run -> realdata_results.json
    python honest_addendum.py       # negative result + safe-sampler check -> honest_results.json
    python adaptive_exogenous.py    # grip-mode adaptive scheme -> adaptive_exogenous_results.json
    python utility_benchmark.py     # company-utility vs privacy sweep -> utility_benchmark_results.json
    python gazzari_style_attacker.py# stronger temporal attacker -> gazzari_style_results.json
    python robustness_study.py      # 45-run consistency -> robustness_results.json
    python hardening_fixes.py       # 4 hardening fixes -> hardening_results.json
    python sanity_checks.py         # 4 assertions (should all PASS)

## Requirements
    pip install -r requirements.txt   # numpy, scipy, scikit-learn, matplotlib

## Modules
    config.py             constants (pre-registered)
    emg_model.py          signal generation + features (RMS, LFA, HFV) + tendon vector
    mechanism.py          risk score, budget, perturbation, w-event odometer
    safe_sampler.py       discrete-Laplace (exact, finite-precision safe) noise
    adversary.py          attackers (gradient-boosted + logistic) + legitimate utility
    hand_model.py, keylogger.py    support (kinematics, keystroke timing)
    experiments.py, realdata_experiment.py, honest_addendum.py,
    adaptive_exogenous.py, utility_benchmark.py, gazzari_style_attacker.py,
    robustness_study.py, hardening_fixes.py, sanity_checks.py    drivers

## Company-utility benchmark (utility_benchmark.py)
The manufacturer's legitimate task (grip-state recognition + aggregate per-finger
estimation) is measured with the standard sEMG classification pipeline (LDA +
Random Forest) on the noised telemetry, across an epsilon sweep with 8 seeds. The
keystroke attack AUC is measured on the same releases. Figures land in `figures/`
(UTIL_curves, UTIL_tradeoff, UTIL_aggregate, UTIL_exogenous).

## Note
This is a simulation-only study; results on real hardware and on the raw
myo-keylogging dataset are the stated next step. See the LICENSE file for terms.
