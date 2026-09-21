# What each switch in this PR is actually worth

The PR argues each change on correctness. This file reports what each one does to the labels and
to a downstream model, because "is it correct" and "is it worth adopting" are different questions.

Evaluation target throughout is `human_labels_2020_2024`, CTO's own annotation. Unless stated
otherwise: downstream model trained on trials with primary completion <= 2019, scored on annotated
trials only, held-out rows only, isotonic recalibration fitted on trials completing in 2020 and
scoring on 2021 onward. Intervals are bootstrap over evaluation rows.

**Read the stratification note first.** It changes how every other number here should be read.

---

## 0. Stratify by status, or the numbers mean something other than they appear to

Every terminated, withdrawn and suspended trial in `human_labels_2020_2024` is annotated failure,
7,142 of 7,142 with no exceptions, and 91.6% of completed trials are annotated success. "Not
completed means failure" reproduces 97.05% of the annotation.

So an unstratified AUROC on this annotation is mostly a score for predicting `overall_status`. That
is not a criticism of the annotation, which is doing its job; it is a statement about what a
benchmark number computed on it measures.

The consequence is concrete. Inside COMPLETED, where status is constant and therefore carries no
information:

| stratum | what the status labeling function scores |
|---|---|
| COMPLETED only | **AUROC 0.5000** — in all three phases, at three temporal split boundaries |
| stopped only | annotation has no variance; every such trial is labelled failure |

Everything below is therefore reported on the COMPLETED stratum as well as overall. We applied this
test to our own work first, and it removed most of a result we had been pleased with: a
feature-rich model that gained +0.060 / +0.060 / +0.076 overall gained -0.023 / +0.043 / -0.031
inside COMPLETED, every interval crossing zero.

---

## 1. `--override_mode vote` — the largest effect in this PR

`legacy` applies the status and p-value overrides on top of the fitted label model. Wherever status
fires, the other channels are discarded.

Measured: status is worth **0.5000** inside COMPLETED, and the channels it overrides are worth a
great deal more. Fitting an aggregation over the distinct non-status channels, on annotated trials
completing 2020-2021 and scoring on 2022 onward:

| phase | released aggregation | learned over non-status channels | delta (95% CI) |
|---|---|---|---|
| 1 | 0.6830 | 0.7949 | +0.113 [0.025, 0.199] |
| 2 | 0.6240 | 0.7668 | +0.144 [0.063, 0.225] |
| 3 | 0.6416 | 0.8011 | +0.160 [0.086, 0.233] |

Checks run before believing this: refitting on shuffled training labels gives 0.469 / 0.474 /
0.509; a model given only the status channel gives 0.5000 in all nine phase-by-boundary cells;
deltas stay positive with intervals clear of zero at the 2020 and 2021 boundaries and stay positive
at 2022, where the test slice falls to 120-224 rows and two of three lose significance on power.
383 annotated trials already reach 0.781 in phase 1; 11,012 are available.

**Caveat, stated because it matters.** The released aggregation is unsupervised and this comparison
is not, so it is not like-for-like and should not be quoted as if it were. The defensible reading is
narrower: the signal is present in the channels the override discards. It also concerns labelling
rather than forecasting, since these votes use post-completion evidence.

---

## 2. `--lf_duplication dedupe`

`legacy` inserts the same labeling function object several times:

    known_lfs_list = [hint_lf, hint_lf, hint_lf, status_lf, status_lf,
                      gpt_lf, gpt_lf, linkage_lf, linkage_lf, ...]

22 columns, 16 distinct channels. Verified byte-identical in all three released matrices:
`hint_train == hint_train2 == hint_train3`, `status == status2`, `gpt == gpt2`,
`linkage == linkage2`, agreement 1.0000 in every case.

`LabelModel` estimates each LF's accuracy from the agreement structure and assumes conditional
independence given the true label. Exact copies are maximally dependent and it cannot tell they are
the same channel, so it reads perfect agreement as corroboration. HINT enters with triple weight and
linkage with double, while `results_reported`, the only channel with 100% coverage, enters with
single. If the intent was to weight those channels up, that is a weight rather than a copied column,
and it should be applied somewhere the independence assumption is not load-bearing.

Measured: on its own, small. Labels move 1.2% in phase 2 and 3.8% in phase 3, and the downstream
effect is not significant in any phase. Combined with the abstention fixes it gives the only
significant COMPLETED-stratum gain among the label-level changes, phase 2 +0.0497 [0.0024, 0.0962];
phase 3 is negative.

**Worth taking because the layout is wrong, not because it moves the benchmark.**

---

## 3. The abstention fixes (D1 linkage, D3 results_reported, D4 gpt)

Under `--label_mode RF` these change **nothing**, and the reason is the useful part. The released
binary label in `phase{1,2,3}_CTO_rf.csv` is reproduced 100.0000% across all 131,160 rows by a
single rule: if `overall_status` is set to a stop reason, failure, otherwise success. `pred_proba`
never falls below 0.5 anywhere, with minima 0.5588, 0.5300 and 0.5375, so the label model never
produces a failure and every failure in the released files came from the status override. No change
to linkage, gpt or results_reported can move a label that status alone determines.

Under DP, which `lfs.py` defaults to, the same three fixes move 25.9% / 20.3% / 17.4% of labels.

Downstream, out of sample, they are worth +0.021 / +0.012 / +0.017 AUROC on the annotation. Inside
COMPLETED that becomes -0.023 / +0.043 / -0.031, every interval crossing zero.

**Take these because the failure votes are unjustified — a trial with no posted results has not
reported a failure, it has not reported — not because the benchmark improves.**

One honest caveat: under DP the correction overshoots. Making `results_reported` abstain removes the
highest-coverage channel almost entirely, and the resulting phase 3 success rate of 0.601 sits above
a plausible true rate nearer 0.44-0.46. Censoring the undecidable rows instead of relabelling them
keeps the base rate closer (0.498 against 0.602 in phase 2) and scores better on Brier and ECE,
though it buys less discrimination.

---

## 4. `--class_balance`

Exposed rather than changed; the released default `.4,.4,.4` remains.

It deserves exposure because results computed at a single fixed value can be entirely an artifact of
that value. On a small verified set of 18 trials, sweeping the prior moves the **released** labels'
own downstream score from 0.386 to 0.864, and reverses the sign of a shipped-versus-corrected
comparison at four of five grid points. Any conclusion drawn at one fixed prior should be swept
before it is reported.

The observed success share also moves by more than 0.25 across 2014-2023, so a single prior reused
for all three phases and all years is a strong assumption.

---

## 5. `--hint_splits`

Exposed rather than changed; the released default includes `test`.

In RF, SVM and LR modes the label model is supervised by the HINT labeling function, so including
TOP's held-out test split means those labels enter the label model. That makes any subsequent
HINT-versus-CTO comparison partly circular. `--hint_splits train,valid` holds them out.

---

## Reproducing

Scripts and per-experiment JSON live outside this repository, in the audit that produced these
numbers. The claims here that can be checked against this repository alone are:

- the duplicate-column claim: read `known_lfs_list` above, or compare the columns in any released
  matrix pairwise;
- the 100.0000% status reproduction: apply the one-line rule to `phase{1,2,3}_CTO_rf.csv`;
- `pred_proba` minima of 0.5588 / 0.5300 / 0.5375: read the released files.
