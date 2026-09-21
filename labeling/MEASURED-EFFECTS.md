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

### On the overshoot, which we raised and then had to withdraw

We previously flagged that this correction overshoots, on the grounds that it lifts the phase 3
success rate to 0.601 against a plausible recent rate nearer 0.44-0.46. That comparison was not
like-for-like and the objection does not survive it. 0.601 is the label rate over every trial in
the matrix, across all years; 0.44-0.46 is an estimate for recent trials only.

Restricted to the cohort the annotation actually covers, against the annotation's own success rate
on the same trials:

| phase | annotation | shipped labels | with this correction | gap |
|---|---|---|---|---|
| 1 | 0.1925 | 0.1442 | 0.0990 | **-0.094** |
| 2 | 0.2951 | 0.2520 | 0.3045 | +0.009 |
| 3 | 0.5233 | 0.4654 | 0.5223 | -0.001 |

In phases 2 and 3 the corrected labels land on the annotation's own base rate, and closer to it than
the shipped labels do. There is no overshoot to answer for there.

Phase 1 is a real problem and goes the other way: the correction **undershoots** by 0.094. It is of a
piece with everything else we saw in phase 1, where the DP labels are unstable — refitting on the
patched matrix moves thousands of labels *against* the direction of the patch, which D1/D3/D4 cannot
do by themselves since they only ever convert a failure vote to an abstention. We would not rely on
phase 1 DP labels without looking at that first.

We also tested a time-aware version of this fix, abstaining only inside the window where a posting
could not yet be overdue under FDAAA 801, on the theory that a long-overdue missing posting is
informative. It does not hold up: 12, 24 and 36 month windows all match the annotation's base rate
worse than unconditional abstention in phases 2 and 3, and score worse downstream.

Censoring the undecidable rows rather than relabelling them remains a reasonable alternative. It
scores better on Brier and ECE than relabelling but buys less discrimination.

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
