# Phase I: what the label measures, and what that does to HINT

Phase I is where HINT is weakest: 0.576 ROC-AUC on TOP as published, 0.564 in our reproduction from the repo. This note
looks at why. We then tested the obvious fix in one pre-registered run, on TOP test and on CTO's own annotation.

**Short version.** A Phase I failure is mostly a trial that stopped for operational reasons, and who sponsors the trial
predicts that well on TOP. Giving HINT the registration fields lifts its TOP Phase I score by +0.077, but the gain does not
hold on CTO's annotation (-0.051), so it is a property of TOP's failure mix, not a fix for HINT. What does hold on both is
smaller and more boring: HINT's released criteria embeddings are empty, and rebuilding them helps in Phases I and II.

---

## 1. A Phase I failure is mostly a stopped trial, and mostly not for drug reasons

TOP, Phase I train+valid (1,161 trials, success rate 0.568): 332 of the 502 failures are terminated, withdrawn or
suspended, and 299 of the 313 terminated trials are labelled failure.

CTO's released RF labels, drug and biologic Phase 1 (41,504 trials, success rate 0.827): every one of the 7,176 failures
is a stopped or withdrawn trial, which follows from the status rule in `MEASURED-EFFECTS.md` section 3. By the reason
recorded in `why_stopped`:

| Phase 1 failures, CTO RF labels | share |
|---|---|
| withdrawn before enrolling | 28.2% |
| stopped, other or no reason given | 27.7% |
| stopped for accrual | 19.0% |
| stopped for business or funding reasons | 15.4% |
| stopped for efficacy or safety | 9.7% |

So at most about one Phase 1 failure in ten is the drug failing. The reasons are keyword-classified, so treat the split as
approximate.

---

## 2. Who runs the trial predicts whether it stops

| | industry share | success, industry | success, other | AUROC of the industry bit alone |
|---|---|---|---|---|
| TOP Ph I train+valid | 54.1% | 0.734 | 0.371 | **0.683** |
| CTO RF labels, Ph 1 | 60.4% | 0.870 | 0.762 | 0.590 |
| CTO RF labels, Ph 2 | 43.2% | 0.782 | 0.742 | 0.527 |
| CTO RF labels, Ph 3 | 62.7% | 0.858 | 0.804 | 0.546 |

On TOP, one unfitted bit (lead sponsor class is INDUSTRY) scores 0.683 on Phase I trials, against HINT's 0.564 to 0.576 on
the test split. The effect is real on CTO's labels too, but much weaker.

The direction depends on why the trial stopped. In CTO Phase 1, industry leads 27.6% of accrual stops and 33.1% of
withdrawals, but 68.9% of business stops and 71.8% of efficacy or safety stops (63.5% of completed trials). So the sponsor
signal separates operational stops from successes and points the wrong way on the rest. LINT already gives the model
the sponsor as text, so "sponsor matters" is not new. What we did not expect is how much of TOP Phase I it explains, and
that it does not transfer (section 4).

---

## 3. HINT's released criteria embeddings are empty

`data/sentence2embedding.pkl` at 8dc0497 is 6 bytes, an empty dict. `protocol2feature` silently substitutes
`torch.zeros(1, 768)` for any sentence missing from the dict:

    inclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in inclusion_criteria if sentence in sentence_2_vec]
    if inclusion_feature == []:
        inclusion_feature = torch.zeros(1,768)

So every run from the repo trains and predicts with an all-zero eligibility-criteria input, and nothing errors. The
published numbers were presumably computed with the real file, which is large (397,398 sentences x 768 floats). We
rebuilt it with HINT's own recipe (BioBERT v1.1, mean of last-layer token vectors, HINT's own sentence splitting) and
retrained HINT with the upstream settings, 3 seeds:

| HINT, rebuilt minus empty embeddings | Ph I | Ph II | Ph III |
|---|---|---|---|
| TOP test | +0.009 [0.001, 0.018] | +0.022 [0.011, 0.034] | +0.010 [-0.001, 0.020] |
| CTO annotation (HINT-featurizable subset) | +0.023 [0.007, 0.040] | +0.025 [0.007, 0.042] | -0.009 [-0.025, 0.006] |

Small, but positive with intervals clear of zero in Phases I and II on both sets. It is a fix to the reference
implementation, and every reproduction from the repo, ours included, has been running without it.

---

## 4. Pre-registered test: does giving HINT the registration fields fix Phase I?

We froze the protocol (arms, seeds, metrics, and the rule for calling it) before scoring anything. Arms, all HINT with
the upstream settings and the rebuilt embeddings, 3 seeds averaged: as is; plus a small branch on 84 standardized
registration fields (sponsor class and collaborators, design and masking, arms, intervention types, outcome counts,
eligibility, data monitoring, title keywords, missing-value flags; no enrollment) added to HINT's logit and trained end
to end; plus the same branch on sponsor class only. Two non-HINT baselines: a logistic
regression on sponsor class, and gradient boosting on the registration fields. Sets: TOP test (627 Phase I trials), and
the trials in `human_labels_2020_2024` that HINT can featurize (494 Phase I). The rule: the claim holds only if the
branch beats HINT on TOP Phase I by at least +0.05 **and** the gain replicates on the annotation. Disclosure: this was our
third scoring of TOP test.

| ROC-AUC, Phase I | TOP test | CTO annotation |
|---|---|---|
| HINT, repo as released | 0.572 | 0.630 |
| HINT, rebuilt embeddings | 0.582 | **0.653** |
| + registration-field branch | **0.658** | 0.602 |
| + sponsor-only branch | 0.656 | 0.609 |
| sponsor-only logistic regression | 0.648 | 0.557 |
| gradient boosting on registration fields | 0.673 | 0.585 |
| **branch minus HINT (rebuilt)** | **+0.077 [0.036, 0.116]** | **-0.051 [-0.113, 0.012]** |

**How the branch is attached.** HINT is retrained from scratch for every arm, on TOP's standard train split
(`data/phase_*_train.csv`), with the upstream per-phase settings (`learn_phase*.py` at 8dc0497: 5 epochs, batch 32,
lr 1e-3 / 3e-4 / 1e-3 for Phases I / II / III, ADMET-pretrained MPNN loaded fresh per phase, last-epoch model kept).
Nothing is added to an already trained model. HINT's own inputs (SMILES, ICD codes, criteria text) and its code are
unchanged. The extra input is one row of 84 numbers per trial, read from a CSV keyed by NCT ID: the registration fields
above, each standardized with the mean and standard deviation of TOP train+valid (features only, no labels), gaps set to
0, plus a missing flag for the 7 fields that have gaps. A subclass of `HINTModel` passes that row through
`Linear(84, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)` and adds the result to HINT's output logit before the sigmoid
and the BCE loss. The last layer is initialized to zero, so at step 0 the model is exactly HINT. The branch trains
jointly with the rest of HINT, same optimizer and learning rate. The sponsor-only arm is the same branch on the 5
sponsor-class columns. Predictions are made in eval mode.

It passes on TOP and fails on the annotation, where the point estimate is negative. By the rule we froze, that makes it a
TOP-specific result. Every seed agrees on both sets.

The failure-type strata (each failure type against all successes) show why:

| Phase I failures | TOP test: n, HINT -> + branch | CTO annotation: n, HINT -> + branch |
|---|---|---|
| stopped for accrual | 113, 0.633 -> 0.771 | 80, 0.680 -> 0.731 |
| withdrawn | 44, 0.644 -> 0.772 | 93, 0.634 -> 0.707 |
| stopped, business | 2 | 90, 0.656 -> 0.472 |
| stopped, efficacy or safety | 28, 0.504 -> 0.398 | 50, 0.648 -> 0.448 |

The branch does the same thing on both sets. It finds the operational stops, and it misranks business and efficacy
stops, because those are mostly industry trials. TOP's Phase I failures are mostly the first kind. The annotation's
include 140 of the second kind out of 400. The branch did replicate in Phase III (+0.054 TOP, +0.090 annotation).

---

## 5. What we would take from this

- **A TOP Phase I number is mostly a score for predicting operational stops.** A one-bit sponsor baseline beats HINT
  there. Reporting Phase I next to a sponsor-only baseline, and split by stop reason, would show what a gain is made of.
- **TOP and the annotation reward different things in Phase I**, because their failure mixes differ. A method tuned on
  TOP Phase I can lose on CTO's labels, as this one did. Worth saying in the benchmark documentation.
- **Ship the criteria embeddings with HINT, or a script that rebuilds them, and make the loader fail on an empty file.**

---

## Reproducing

Scripts, predictions and the frozen protocol live outside this repository, in the audit that produced these numbers.
Checkable against this repository and the HINT repo alone:

- the empty embeddings file: `data/sentence2embedding.pkl` at HINT 8dc0497 is 6 bytes, and the zeros fallback is in
  `HINT/protocol_encode.py`, `protocol2feature`;
- the Phase 1 failure composition: join `phase1_CTO_rf.csv` to CTTI `overall_status` and `why_stopped`;
- the industry bit: CTTI `source_class` on the same rows.
