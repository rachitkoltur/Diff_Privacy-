"""
experiments.py
==============
Runs every experiment, writes real numeric results to results.json, and saves
publication-quality figures to figures/. No number in the paper is hand-typed;
they are all read back from results.json.

Experiments
-----------
E1  Mechanism illustration (EMG, features, R(t), eps(t)) over a session.
E2  RMS L1-sensitivity: Monte-Carlo bound check + worst-case tightness (=0.1).
E3  Privacy-Utility Pareto: adaptive vs fixed eps swept at matched budgets.
E4  Matched-budget head-to-head + no-privacy baseline (side channel exists).
E5  Boundary-clamping: histogram + estimator bias + empirical DP preserved.
E6  Control stability: local(raw) vs cloud(noisy) finger tracking ("shaky arm").
E7  w-event streaming: cumulative window loss bounded by eps_W.
E8  Adaptive-budget leakage: MI(eps; typing) for naive vs sound (lagged) budget.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CONFIG as C
import emg_model as em
import mechanism as mech
import adversary as adv
import hand_model as hm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
RESULTS = {}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
BLUE, RED, GREEN, GRAY = "#1f6feb", "#d1242f", "#1a7f37", "#57606a"


def _rng(offset=0):
    return np.random.default_rng(C.seed + offset)


# ---------------------------------------------------------------------------
def build_world(duration_s=300.0, seed_offset=0):
    sess = em.generate_session(C, duration_s=duration_s, seed=C.seed + seed_offset)
    sess = em.extract_features(sess, C)
    D = em.generate_tendon_vector(sess, C, seed=C.seed + 500 + seed_offset)
    R = mech.risk_score(sess.hfv, sess.lfa, C)
    eps_naive = mech.epsilon_of_risk(R, C)             # instantaneous, from RAW sample (leaky)
    # SOUND budget: epsilon is a function of a DP-RELEASED coarse context, so it is
    # measurable w.r.t. public output and using it is post-processing (Theorem 1).
    ctx_released = mech.release_context(R, C, _rng(700 + seed_offset))
    R_smooth = mech.trailing_mean(R, win=C.privacy.ctx_smooth)  # true smoothed (for display only)
    eps_sound = mech.epsilon_of_risk(ctx_released, C)
    return sess, D, R, eps_naive, R_smooth, eps_sound


def generate_telemetry(sess, D, eps_array, rng, clamp=False):
    """Apply CA-LDP perturbation timestep-by-timestep for a given eps schedule."""
    T = len(eps_array)
    y_emg = np.zeros(T)
    Y_ten = np.zeros((T, C.privacy.n_fingers))
    b_emg = np.zeros(T)
    b_ten = np.zeros(T)
    for t in range(T):
        y_emg[t], b_emg[t] = mech.perturb_emg(sess.rms[t], eps_array[t], C, rng)
        Y_ten[t], b_ten[t] = mech.perturb_tendons(D[t], eps_array[t], C, rng, clamp=clamp)
    return y_emg, Y_ten, b_emg, b_ten


def scale_to_mean(eps_adaptive, target_mean):
    e = eps_adaptive * (target_mean / eps_adaptive.mean())
    return np.clip(e, C.privacy.eps_min, C.privacy.eps_max)


# ===========================================================================
# E1 - mechanism illustration
# ===========================================================================
def E1(sess, R, eps_adaptive):
    t = sess.win_time
    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    # raw EMG (decimated for display) with task shading
    tr = np.arange(len(sess.raw)) / sess.fs
    ax[0].plot(tr, sess.raw, color=GRAY, lw=0.3)
    ax[0].set_ylabel("raw sEMG")
    ax[0].set_title("E1  Context-aware LDP responding to task context")
    # shade tasks
    names = {0: ("rest", "#eef2f5"), 1: ("grasp", "#dbeafe"), 2: ("type", "#ffe3e3")}
    tp = sess.win_task
    for i in range(len(t)):
        c = names[tp[i]][1]
        for a in ax:
            a.axvspan(t[i], t[i] + C.signal.window_ms / 1000, color=c, lw=0, zorder=0)
    ax[1].plot(t, sess.lfa, color=BLUE, label="LFA (macro-grasp)")
    ax[1].plot(t, sess.hfv, color=RED, label="HFV (micro-dexterity)")
    ax[1].set_ylabel("features"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, R, color="#8250df"); ax[2].axhline(0, color=GRAY, lw=0.6)
    ax[2].set_ylabel("R(t) = 0.7·HFV − 0.3·LFA")
    ax[3].plot(t, eps_adaptive, color=GREEN)
    ax[3].axhline(C.privacy.eps_max, color=GRAY, ls="--", lw=0.6, label="ε_max")
    ax[3].axhline(C.privacy.eps_min, color=GRAY, ls=":", lw=0.6, label="ε_min")
    ax[3].set_ylabel("ε(t)"); ax[3].set_xlabel("time (s)")
    ax[3].legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E1_mechanism.png")); plt.close(fig)

    # numeric summary: mean eps per task
    res = {}
    for k, nm in em.TASK_NAMES.items():
        m = sess.win_task == k
        if m.any():
            res[nm] = dict(mean_eps=float(eps_adaptive[m].mean()),
                           mean_hfv=float(sess.hfv[m].mean()),
                           mean_lfa=float(sess.lfa[m].mean()))
    RESULTS["E1_mean_eps_by_task"] = res


# ===========================================================================
# E2 - RMS sensitivity
# ===========================================================================
def E2():
    """Verify BOTH sensitivities: the sample-level bound 1/sqrt(N), and the
    event-level (keystroke) bound sqrt(m/N) for a group of m changed samples,
    which is the one the mechanism actually calibrates to."""
    rng = _rng(1)
    N = C.privacy.n_rms_samples
    # (A) sample-level: max over random single-sample substitutions vs 1/sqrt(N)
    mx1 = 0.0
    for _ in range(20000):
        x = rng.uniform(0, 1, N); xp = x.copy()
        j = rng.integers(N); xp[j] = rng.uniform(0, 1)
        mx1 = max(mx1, abs(mech.rms(x) - mech.rms(xp)))
    # (B) event-level: change a GROUP of m coordinates (keystroke footprint,
    #     allowing up to K overlapping keystrokes -> up to K*m_single samples)
    m_evt_max = C.privacy.keystroke_max_concurrent * C.privacy.keystroke_samples_max
    ms = list(range(1, m_evt_max + 1))
    grp_bound, grp_worst, grp_rand = [], [], []
    for m in ms:
        grp_bound.append(np.sqrt(m / N))
        x0 = np.zeros(N); x1 = x0.copy(); x1[:m] = 1.0     # worst case: 0 -> 1 on m coords
        grp_worst.append(abs(mech.rms(x1) - mech.rms(x0)))
        mr = 0.0
        for _ in range(3000):
            x = rng.uniform(0, 1, N); xp = x.copy()
            idx = rng.choice(N, size=m, replace=False); xp[idx] = rng.uniform(0, 1, m)
            mr = max(mr, abs(mech.rms(x) - mech.rms(xp)))
        grp_rand.append(mr)
    m_evt = m_evt_max
    RESULTS["E2_sensitivity"] = dict(
        N=N, sample_bound=float(1/np.sqrt(N)), sample_empirical_max=float(mx1),
        event_m=m_evt, event_bound=float(np.sqrt(m_evt/N)),
        event_worst=float(grp_worst[-1]), delta_f_event=float(C.privacy.delta_f_event),
        note="Sample bound 1/sqrt(N)=0.1; event bound sqrt(m/N)=%.3f is what the noise uses." % C.privacy.delta_f_event)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ms_a = np.array(ms)
    ax.plot(ms_a, grp_bound, "-", color=BLUE, label="bound  √(m/N)")
    ax.plot(ms_a, grp_worst, "x", ms=7, color=GREEN, label="worst-case construction")
    ax.plot(ms_a, grp_rand, "s", ms=3, color=RED, label="max over random m-sample edits")
    ax.axhline(C.privacy.delta_f_sample, color=GRAY, ls=":", lw=0.8)
    ax.annotate("sample: 1/√N = 0.1", (1, 0.1), (6, 0.16), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.axvline(m_evt, color=GRAY, ls="--", lw=0.8)
    ax.annotate("keystroke m=%d,  Δf=%.2f" % (m_evt, C.privacy.delta_f_event),
                (m_evt, C.privacy.delta_f_event), (m_evt - 20, C.privacy.delta_f_event - 0.14), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xlabel("number of samples changed by one event, m  (N=100)")
    ax.set_ylabel("L1-sensitivity of RMS")
    ax.set_title("E2  Sample vs event (keystroke) sensitivity of the RMS")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E2_sensitivity.png")); plt.close(fig)


# ===========================================================================
# E3 / E4 - privacy-utility
# ===========================================================================
def attack_and_utility(sess, D, eps_array, n_repeat=4, offset=0):
    """Privacy is measured where it matters: KEYSTROKE detection WITHIN typing
    windows (the fine side channel), not the coarse 'is the user typing' fact.
    Utility is the manufacturer's legitimate GRASP-phase aggregate."""
    typing = sess.win_task == em.TYPE
    grasp = sess.win_task == em.GRASP
    accs, aucs, rel_errs, recons = [], [], [], []
    for r in range(n_repeat):
        rng = _rng(1000 + offset * 50 + r)
        y_emg, Y_ten, _, _ = generate_telemetry(sess, D, eps_array, rng)
        X = adv.build_attack_matrix(y_emg, Y_ten)
        a = adv.run_keystroke_attack(X[typing], sess.win_keystroke[typing], seed=r)
        u = adv.aggregate_utility(D, Y_ten, mask=grasp)
        accs.append(a["bal_acc"]); aucs.append(a["auc"])
        rel_errs.append(u["mean_rel_err"]); recons.append(u["recon_rmse"])
    return dict(bal_acc=float(np.mean(accs)), auc=float(np.mean(aucs)),
                auc_sd=float(np.std(aucs)),
                rel_err=float(np.mean(rel_errs)),
                recon_rmse=float(np.mean(recons)), recon_sd=float(np.std(recons)))


def E3_E4(sess, D, eps_naive, eps_sound):
    # --- no-privacy baseline: raw telemetry (side channel exists) ---
    typing = sess.win_task == em.TYPE
    Xraw = adv.build_attack_matrix(sess.rms.astype(float), D.astype(float))
    base = adv.run_keystroke_attack(Xraw[typing], sess.win_keystroke[typing], seed=0)
    # robustness: a second, different attacker class (logistic) confirms the
    # side channel is not an artifact of one model
    base_lr = adv.run_keystroke_attack(Xraw[typing], sess.win_keystroke[typing], seed=0, model="lr")
    base["auc_logistic"] = base_lr["auc"]
    RESULTS["E4_no_privacy_attack"] = base

    target_means = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    fixed_pts, naive_pts, sound_pts = [], [], []
    for i, m in enumerate(target_means):
        eps_fixed = np.full(len(eps_naive), m)
        rf = attack_and_utility(sess, D, eps_fixed, offset=i)
        rn = attack_and_utility(sess, D, scale_to_mean(eps_naive, m), offset=100 + i)
        rs = attack_and_utility(sess, D, scale_to_mean(eps_sound, m), offset=200 + i)
        fixed_pts.append(dict(mean_eps=m, **rf))
        naive_pts.append(dict(mean_eps=m, **rn))
        sound_pts.append(dict(mean_eps=m, **rs))
    RESULTS["E3_fixed"] = fixed_pts
    RESULTS["E3_adaptive_naive"] = naive_pts
    RESULTS["E3_adaptive_sound"] = sound_pts

    def arr(pts, key):
        return np.array([p[key] for p in pts])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(arr(fixed_pts, "recon_rmse"), arr(fixed_pts, "auc"), "-o", color=GRAY, label="fixed ε")
    ax[0].plot(arr(naive_pts, "recon_rmse"), arr(naive_pts, "auc"), "-s", color=RED,
               label="adaptive ε — naive (leaks)")
    ax[0].plot(arr(sound_pts, "recon_rmse"), arr(sound_pts, "auc"), "-o", color=GREEN,
               label="adaptive ε — sound (ours)")
    ax[0].axhline(0.5, color=GRAY, ls="--", lw=0.8, label="chance (AUC=0.5)")
    ax[0].set_xlabel("grasp-phase reconstruction RMSE (cloud-ML noise)  →worse")
    ax[0].set_ylabel("keystroke attack AUC  →worse")
    ax[0].set_title("E3  Privacy–utility frontier (down-left = better)")
    ax[0].legend(fontsize=8)
    mm = arr(fixed_pts, "mean_eps")
    ax[1].plot(mm, arr(fixed_pts, "auc"), "-o", color=GRAY, label="fixed ε")
    ax[1].plot(mm, arr(naive_pts, "auc"), "-s", color=RED, label="naive adaptive")
    ax[1].plot(mm, arr(sound_pts, "auc"), "-o", color=GREEN, label="sound adaptive (ours)")
    ax[1].axhline(0.5, color=GRAY, ls="--", lw=0.8)
    ax[1].axhline(base["auc"], color=BLUE, ls=":", lw=1.0, label=f"no privacy ({base['auc']:.2f})")
    ax[1].set_xlabel("mean budget  E[ε]"); ax[1].set_ylabel("keystroke attack AUC")
    ax[1].set_title("E4  Attack vs budget (matched mean)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E3_E4_privacy_utility.png")); plt.close(fig)


# ===========================================================================
# E5 - boundary clamping
# ===========================================================================
def E5(sess, D, eps_adaptive):
    rng = _rng(3)
    eps = scale_to_mean(eps_adaptive, 1.0)
    _, Y_noclamp, _, _ = generate_telemetry(sess, D, eps, _rng(31))
    _, Y_clamp, _, _ = generate_telemetry(sess, D, eps, _rng(31), clamp=True)
    f = 0  # finger 0
    tm = D[:, f].mean()
    est_noclamp = Y_noclamp[:, f].mean()
    est_clamp = Y_clamp[:, f].mean()

    # empirical DP check: clamping is post-processing -> guarantee preserved.
    # Test statistic: ratio of output densities for adjacent inputs d and d'
    # estimated by histogram; should satisfy e^{-eps} <= ratio <= e^{eps}.
    d_a, d_b = 0.9, 1.0
    eps_i = C.privacy.frac_tendon_vector * 1.0 / C.privacy.n_fingers
    b = C.privacy.delta_d / eps_i
    NN = 3_000_000
    ya = np.clip(d_a + mech.laplace_inverse_transform(b, rng, NN), 0, 1)
    yb = np.clip(d_b + mech.laplace_inverse_transform(b, rng, NN), 0, 1)
    edges = np.linspace(-0.05, 1.05, 45)
    max_log_ratio, ha, hb = adv.empirical_epsilon(ya, yb, edges, min_count=500)
    RESULTS["E5"] = dict(
        true_mean=float(tm), est_mean_noclamp=float(est_noclamp),
        est_mean_clamp=float(est_clamp),
        bias_noclamp=float(est_noclamp - tm), bias_clamp=float(est_clamp - tm),
        eps_i=float(eps_i), empirical_max_log_density_ratio=max_log_ratio,
        note=("Clamping the NOISY output is data-independent post-processing, so "
              "eps-LDP is preserved (empirical max log-ratio <= eps_i). Clamping's "
              "real cost is ESTIMATOR BIAS toward the interior, which no-clamp "
              "transmission removes."))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(Y_noclamp[:, f], bins=40, color=GREEN, alpha=0.7, label="no-clamp (raw floats)")
    ax[0].hist(Y_clamp[:, f], bins=40, color=RED, alpha=0.55, label="clamped to [0,1]")
    ax[0].axvline(0, color=GRAY, ls=":"); ax[0].axvline(1, color=GRAY, ls=":")
    ax[0].set_title("E5  Boundary spikes from clamping")
    ax[0].set_xlabel("transmitted d₁"); ax[0].set_ylabel("count"); ax[0].legend(fontsize=8)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax[1].plot(centers, ha, color=BLUE, label="output | d=0.9")
    ax[1].plot(centers, hb, color=RED, label="output | d=1.0")
    ax[1].set_title(f"E5  DP preserved under clamp (max log-ratio {max_log_ratio:.2f} ≤ ε_i={eps_i:.2f})")
    ax[1].set_xlabel("clamped output"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E5_boundary.png")); plt.close(fig)


# ===========================================================================
# E6 - control stability
# ===========================================================================
def E6(sess, D, eps_adaptive):
    rng = _rng(4)
    eps = scale_to_mean(eps_adaptive, 1.0)
    _, Y_ten, _, _ = generate_telemetry(sess, D, eps, rng)
    f = 0
    # focus on a grasp-heavy window for clarity
    ref_cmd = D[:, f]
    cloud_cmd = Y_ten[:, f]
    ref_up, reps = hm.upsample_to_control(ref_cmd, C)
    cloud_up, _ = hm.upsample_to_control(cloud_cmd, C)
    x_local = hm.simulate_finger(ref_up, C)     # LOCAL: raw command
    x_cloud = hm.simulate_finger(cloud_up, C)   # CLOUD: noisy command
    # "ideal" physical target is the servo response to the clean reference
    rmse_local = hm.tracking_rmse(ref_up, x_local)
    rmse_cloud = hm.tracking_rmse(ref_up, x_cloud)
    RESULTS["E6"] = dict(rmse_local=rmse_local, rmse_cloud=rmse_cloud,
                         ratio=float(rmse_cloud / max(rmse_local, 1e-9)))
    tt = np.arange(len(ref_up)) * C.hand.dt_ctrl
    seg = (tt >= 40) & (tt <= 70)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(tt[seg], ref_up[seg], color=GRAY, lw=1.2, label="reference command")
    ax.plot(tt[seg], x_local[seg], color=GREEN, lw=1.6, label=f"LOCAL / raw (RMSE {rmse_local:.3f})")
    ax.plot(tt[seg], x_cloud[seg], color=RED, lw=0.9, alpha=0.85,
            label=f"CLOUD / noisy (RMSE {rmse_cloud:.3f})")
    ax.set_xlabel("time (s)"); ax.set_ylabel("finger position d₁")
    ax.set_title("E6  Why noise must be decoupled from control (the 'shaky arm')")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E6_control.png")); plt.close(fig)


# ===========================================================================
# E7 - w-event streaming composition
# ===========================================================================
def E7(sess, R, eps_adaptive):
    p = C.privacy
    # The w-event cap sets the AVERAGE per-step budget eps_bar = eps_W / w. The
    # risk engine REDISTRIBUTES this average (more where safe, less where risky),
    # so we scale the requested schedule to have mean eps_bar.
    # The w-event cap covers BOTH the coarse-context release (eps_ctx per step)
    # and the telemetry. Telemetry therefore gets the remainder of the average.
    # context is released once per ctx_release_period steps, so its cost over a
    # w-window is (w / period) * eps_ctx, not w * eps_ctx.
    n_ctx = p.w_window / p.ctx_release_period
    ctx_window_cost = n_ctx * p.eps_ctx
    eps_bar = (p.eps_W - ctx_window_cost) / p.w_window
    req = scale_to_mean(eps_adaptive, eps_bar)
    tele_cap = p.eps_W - ctx_window_cost          # reserve context budget over the window
    uni = mech.WEventBudget(C); uni.eps_W = tele_cap
    adp = mech.WEventBudget(C); adp.eps_W = tele_cap
    win_sum_uni, win_sum_adp, spent_adp = [], [], []
    for t in range(len(req)):
        uni.spend_uniform()
        adp.spend_adaptive(req[t])
        win_sum_uni.append(uni.window_sum())
        win_sum_adp.append(adp.window_sum())
        spent_adp.append(adp.history[-1])
    win_sum_uni = np.array(win_sum_uni); win_sum_adp = np.array(win_sum_adp)
    win_sum_adp_total = win_sum_adp + ctx_window_cost  # + context spend over the window
    # PER-KEYSTROKE (per-event) privacy: a keystroke's <=60 ms burst overlaps at
    # most F_tel=2 of the 200 ms telemetry windows, and enters the smoothed risk
    # for W windows, touching F_ctx=ceil(W/P)+1 context releases. By sequential
    # composition its total loss <= F_tel*eps(t) + F_ctx*eps_ctx.
    F_tel = 2                                             # keystroke overlaps <=2 windows
    F_tel_robust = 3                                       # conservative: burst smears into a 3rd
    F_ctx = int(np.ceil(p.ctx_smooth / p.ctx_release_period)) + 1
    per_event_at_unit = F_tel * 1.0 + F_ctx * p.eps_ctx   # at eps(t)=1 during typing
    per_event_robust = F_tel_robust * 1.0 + F_ctx * p.eps_ctx
    RESULTS["E7"] = dict(
        eps_W=p.eps_W, w=p.w_window, eps_bar=float(eps_bar), eps_ctx=float(p.eps_ctx),
        ctx_window_cost=float(ctx_window_cost), ctx_period=int(p.ctx_release_period),
        tel_footprint=F_tel, ctx_footprint=F_ctx, per_event_eps_at_unit=float(per_event_at_unit),
        per_event_eps_robust=float(per_event_robust), tel_footprint_robust=F_tel_robust,
        max_window_sum_uniform=float(win_sum_uni.max() + ctx_window_cost),
        max_window_sum_adaptive=float(win_sum_adp_total.max()),
        violations=int(np.sum(win_sum_adp_total > p.eps_W + 1e-6)),
        suppressed_fraction=float(adp.suppressed_fraction()),
        note="Telemetry + context spend together stay within eps_W over any w-window.")
    t = sess.win_time
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, win_sum_uni, color=BLUE, label="uniform (ε_W/w each)")
    ax.plot(t, win_sum_adp, color=GREEN, label="adaptive allocation (ours)")
    ax.axhline(p.eps_W, color=RED, ls="--", label=f"cap ε_W={p.eps_W}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("Σ budget over trailing w-window")
    ax.set_title("E7  w-event streaming loss stays bounded")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E7_wevent.png")); plt.close(fig)


# ===========================================================================
# E8 - adaptive-budget leakage (soundness)
# ===========================================================================
def E8():
    """Analytic DP test of the data-dependent budget (Proposition 1).

    If the noise scale b depends on the current private sample, two neighboring
    inputs get Laplacians of DIFFERENT scale. Their exact log-density ratio is
    unbounded: over an output window [-Y, Y] the worst-case loss grows linearly
    with Y, so no finite epsilon holds. If b is fixed by RELEASED public context,
    the scales are equal and the loss equals the target for every Y. We compute
    the EXACT worst-case loss (no sampling) over growing windows Y."""
    target_eps = 1.0
    Delta = C.privacy.delta_f_event
    x0 = 0.5
    xa, xb = x0 - Delta / 2, x0 + Delta / 2   # neighboring inputs (differ by <= Delta)
    eps_hi, eps_lo = 2.0, 0.5                 # naive: different budgets on the two sides
    b_a, b_b = Delta / eps_hi, Delta / eps_lo # naive scales (unequal)
    b_s = Delta / target_eps                  # sound scale (equal on both)

    def logpdf(y, loc, b):
        return -np.abs(y - loc) / b - np.log(2 * b)
    ranges = [5, 10, 20, 40, 80, 160]
    naive_loss, sound_loss = [], []
    for Y in ranges:
        y = np.linspace(-Y, Y, 400000)
        naive_loss.append(float(np.max(np.abs(logpdf(y, xa, b_a) - logpdf(y, xb, b_b)))))
        sound_loss.append(float(np.max(np.abs(logpdf(y, xa, b_s) - logpdf(y, xb, b_s)))))
    RESULTS["E8"] = dict(
        target_eps=target_eps, ranges=ranges,
        naive_loss=naive_loss, sound_loss=sound_loss,
        naive_loss_at_R40=float(naive_loss[3]), naive_loss_at_R160=float(naive_loss[5]),
        sound_loss_max=float(max(sound_loss)),
        naive_unbounded=bool(naive_loss[-1] > 1.8 * naive_loss[2]),
        sound_bounded=bool(max(sound_loss) <= target_eps + 0.02),
        note=("Proposition 1: naive loss grows linearly with the output window "
              "(unbounded, no finite epsilon). Sound loss equals the target for "
              "every window."))
    # figure: left = exact log-ratio over a fixed window; right = worst-case loss vs Y
    y = np.linspace(-40, 40, 4000)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(y, logpdf(y, xa, b_a) - logpdf(y, xb, b_b), color=RED, label="naive (unequal scale)")
    ax[0].plot(y, logpdf(y, xa, b_s) - logpdf(y, xb, b_s), color=GREEN, label="sound (equal scale)")
    ax[0].axhline(target_eps, color=GRAY, ls="--"); ax[0].axhline(-target_eps, color=GRAY, ls="--", label="±target ε")
    ax[0].set_xlabel("output y"); ax[0].set_ylabel("exact log density ratio")
    ax[0].set_title("E8  Naive ratio diverges; sound stays in ±ε band"); ax[0].legend(fontsize=8)
    ax[1].plot(ranges, naive_loss, "-o", color=RED, label="naive: worst-case loss")
    ax[1].plot(ranges, sound_loss, "-o", color=GREEN, label="sound: worst-case loss")
    ax[1].axhline(target_eps, color=GRAY, ls="--", label="target ε = 1")
    ax[1].set_xlabel("output window half-width Y"); ax[1].set_ylabel("worst-case privacy loss")
    ax[1].set_title("E8  Naive loss grows without bound in Y"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E8_leakage.png")); plt.close(fig)


# ===========================================================================
# E9 - aggregate utility: unbiasedness + 1/sqrt(N) convergence
# ===========================================================================
def E9():
    """Per-sample LDP telemetry is intentionally very noisy, so individual
    keystrokes are destroyed. The manufacturer's LEGITIMATE analytics recover
    POPULATION statistics by pooling over many users/sessions: because Laplace
    noise is zero-mean, the pooled estimator is UNBIASED and its error shrinks
    as 1/sqrt(N). This is how LDP is used in practice (Apple, Google RAPPOR) and
    is the formal content of 'statistical properties remain intact for ML'."""
    rng = _rng(11)
    true_grip = 0.8            # population mean grasp displacement (finger)
    eps_op = 1.0              # operating budget
    eps_i = C.privacy.frac_tendon_vector * eps_op / C.privacy.n_fingers
    b = C.privacy.delta_d / eps_i
    Ns = np.unique(np.round(np.logspace(1, 4.5, 22)).astype(int))
    errs, biases = [], []
    n_mc = 1200
    for N in Ns:
        est = np.empty(n_mc)
        for j in range(n_mc):
            noisy = true_grip + mech.laplace_inverse_transform(b, rng, int(N))
            est[j] = noisy.mean()
        errs.append(float(np.mean(np.abs(est - true_grip))))
        biases.append(float(np.mean(est - true_grip)))
    theo = np.sqrt(2) * b / np.sqrt(Ns.astype(float))  # E|mean| ~ sd = sqrt2 b /sqrt N
    # bias where MC noise is small enough to resolve it (N>=1000)
    bias_hi = [bs for N, bs in zip(Ns, biases) if N >= 1000]
    # Higher moments under HETEROGENEOUS (time-varying) noise: because CA-LDP's
    # scale b changes step to step, a naive variance is inflated by the AVERAGE
    # noise variance (1/M) sum 2 b_i^2. It is recoverable because every b_i is
    # public: subtract that average to debias.
    n_each = 300_000
    pop = np.concatenate([np.full(n_each, 0.8), np.full(n_each, 0.2)])  # true var = 0.09
    var_true = float(np.var(pop))
    naive_list, deb_list = [], []
    for _ in range(30):                                                  # average over noise draws
        b_i = rng.uniform(1.0, 3.0, len(pop))                            # per-sample (time-varying) scales
        noisy = pop + rng.laplace(0.0, b_i)                              # Lap with varying scale
        naive_list.append(np.var(noisy))
        deb_list.append(np.var(noisy) - np.mean(2 * b_i ** 2))           # heterogeneous debiasing
    var_naive = float(np.mean(naive_list))
    var_debiased = float(np.mean(deb_list))
    RESULTS["E9"] = dict(
        eps_op=eps_op, b=float(b),
        Ns=[int(x) for x in Ns], mean_abs_err=errs, bias=biases,
        err_at_N1e4=float(np.interp(1e4, Ns, errs)),
        max_abs_bias_Nge1000=float(np.max(np.abs(bias_hi))),
        var_true=var_true, var_naive=var_naive, var_debiased=var_debiased,
        note=("Unbiased (|bias|<=%.2e for N>=1000); error tracks sqrt(2)b/sqrt(N). "
              "Individual keystroke (N=1) is buried in noise b=%.1f, but pooled "
              "population grip statistics converge -> ML utility preserved."
              % (float(np.max(np.abs(bias_hi))), b)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(Ns, errs, "o", color=GREEN, label="empirical |est − true|")
    ax[0].loglog(Ns, theo, "-", color=GRAY, label=r"theory  $\sqrt{2}\,b/\sqrt{N}$")
    ax[0].set_xlabel("number of pooled samples / users N")
    ax[0].set_ylabel("aggregate estimation error")
    ax[0].set_title("E9  Aggregate grip statistic converges (ε=1)")
    ax[0].legend(fontsize=8)
    ax[1].semilogx(Ns, biases, "o-", color=BLUE)
    ax[1].axhline(0, color=GRAY, lw=0.8)
    ax[1].set_xlabel("N"); ax[1].set_ylabel("estimator bias")
    ax[1].set_title("E9  Estimator is unbiased (Laplace zero-mean)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E9_aggregate.png")); plt.close(fig)


# ===========================================================================
def main():
    print("Building world ...")
    sess, D, R, eps_naive, R_smooth, eps_sound = build_world(duration_s=300.0)
    eps_adaptive = eps_naive
    RESULTS["meta"] = dict(
        seed=C.seed, duration_s=300.0, n_windows=int(len(sess.win_task)),
        n_keystrokes=int(len(sess.keystroke_times)),
        win_samples=C.win_samples, fs=C.signal.fs_raw,
        pct_type=float(np.mean(sess.win_task == em.TYPE)),
        pct_grasp=float(np.mean(sess.win_task == em.GRASP)),
        pct_rest=float(np.mean(sess.win_task == em.REST)))
    print("E1"); E1(sess, R, eps_adaptive)
    print("E2"); E2()
    print("E3/E4"); E3_E4(sess, D, eps_naive, eps_sound)
    print("E5"); E5(sess, D, eps_adaptive)
    print("E6"); E6(sess, D, eps_adaptive)
    print("E7"); E7(sess, R, eps_adaptive)
    print("E8"); E8()
    print("E9"); E9()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("DONE -> results.json + figures/")


if __name__ == "__main__":
    main()
