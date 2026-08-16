"""
gazzari_style_attacker.py
=========================
A STRONGER, TEMPORAL neural-network attacker, in the STYLE of the Gazzari et al.
(2021) deep-learning keylogging attack, tested against the defense.

Important honesty note: this is NOT Gazzari's exact published model. Their attack
is a deep network (CNN/RNN) trained on the raw myo-keylogging recordings; running
that would require their repository, the full raw dataset, and a deep-learning
stack. This module instead builds the closest honest approximation that can run
here: a temporal (sequence) neural network that sees a sliding window of many
consecutive transmitted telemetry timesteps at once, which is what a sequence
model effectively exploits. Against the DEFENDED stream the attacker only ever
observes the noisy transmitted telemetry (noisy RMS + noisy tendon), never the
raw signal, so temporal context is the meaningful way to make it stronger.

Reports attack AUC with no privacy and under the fixed defense.
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import emg_model as em
import mechanism as mech
from config import CONFIG as C

HERE = os.path.dirname(os.path.abspath(__file__))
L = 15  # temporal context length (3 s at 5 Hz), sequence-model style


def telem(sess, D, eps_arr, seed):
    rng = np.random.default_rng(seed)
    T = len(eps_arr); yr = np.zeros(T); YT = np.zeros((T, C.privacy.n_fingers))
    for t in range(T):
        yr[t], _ = mech.perturb_emg(sess.rms[t], eps_arr[t], C, rng)
        YT[t], _ = mech.perturb_tendons(D[t], eps_arr[t], C, rng)
    return yr, YT


def temporal_matrix(yr, YT):
    """Stack the last L timesteps of transmitted telemetry (RMS + 4 tendons)
    into one feature vector per timestep: a sequence-model style input."""
    T = len(yr)
    base = np.column_stack([yr, YT])          # (T, 5)
    feats = np.zeros((T, L * base.shape[1]))
    for t in range(T):
        a = max(0, t - L + 1)
        chunk = base[a:t + 1]
        if len(chunk) < L:
            chunk = np.vstack([np.repeat(chunk[:1], L - len(chunk), axis=0), chunk])
        feats[t] = chunk.flatten()
    return feats


def attack(feats, labels, seed=0):
    if labels.sum() < 10 or (len(labels) - labels.sum()) < 10:
        return 0.5, 0.5
    Xtr, Xte, ytr, yte = train_test_split(feats, labels, test_size=0.4,
                                          random_state=seed, stratify=labels)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=seed)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), float(balanced_accuracy_score(yte, (p >= 0.5).astype(int)))


def main():
    sess = em.generate_session(C, duration_s=300.0, seed=C.seed)
    sess = em.extract_features(sess, C)
    D = em.generate_tendon_vector(sess, C, seed=C.seed + 500)
    typ = sess.win_task == em.TYPE
    lab = sess.win_keystroke

    # no privacy: attacker sees the clean transmitted telemetry
    fn = temporal_matrix(sess.rms.astype(float), D.astype(float))
    np_auc, np_bal = attack(fn[typ], lab[typ])

    # defended at eps_op = 0.5 (fixed)
    yl, YL = telem(sess, D, np.full(len(sess.win_task), 0.5), C.seed + 31)
    fd = temporal_matrix(yl, YL)
    df_auc, df_bal = attack(fd[typ], lab[typ])

    out = dict(model=f"temporal sequence MLP (128,64), context L={L} (~3 s), Gazzari-style",
               is_gazzari_exact_model=False,
               no_privacy_auc=round(np_auc, 3), no_privacy_bal=round(np_bal, 3),
               defended_auc=round(df_auc, 3), defended_bal=round(df_bal, 3),
               note="Stronger temporal attacker approximating a sequence deep model. "
                    "NOT Gazzari's exact trained network; running that needs their repo, "
                    "the full raw dataset, and a deep-learning stack.")
    json.dump(out, open(os.path.join(HERE, "gazzari_style_results.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
