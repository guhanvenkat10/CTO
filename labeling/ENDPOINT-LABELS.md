# Training on primary-endpoint results instead of TOP's label

TOP's label mixes two kinds of failure: trials that stopped for operational reasons, and trials that finished but missed
their primary endpoint. Most completed trials now post their primary-outcome analysis, p-value included, on
ClinicalTrials.gov. This note asks what happens if a model is trained directly on that outcome, and tests it on trials
whose results did not exist when the training data was frozen.

**Short version.** Trained on whether earlier trials' first primary outcome was significant, a registry model forecasts
that outcome on 924 fresh trials at 0.723 ROC-AUC, against 0.688 for the same model trained on TOP's label (+0.035
[0.008, 0.060]). On the 277 of those trials HINT can featurize, TOP-trained HINT scores 0.556 at the same question.
Pre-registered, scored once. Phase I could not be tested, and this is a result on the endpoint question, not a new TOP
benchmark number.

---

## 1. Why the endpoint

On the trials where both exist, TOP's label mostly follows the primary result. Among completed TOP train trials with a
CTO p-value vote, that vote alone ranks TOP's label at 0.81 to 0.86 ROC-AUC by phase. And on a held-out split built to
look like TOP test (fit on trials completed before a cutoff, score trials started after it), about 70% of our registry
model's ranking errors were trials that completed and missed their endpoint. So the remaining errors are mostly the
efficacy question, and TOP's label teaches it only indirectly.

## 2. Data

**Labels for training.** For every training-eligible trial (TOP train+valid, plus non-TOP CTO trials that started before
2014-01-01) with posted results, we took the first primary outcome that reports an analysis p-value, from the
ClinicalTrials.gov v2 API (`resultsSection.outcomeMeasuresModule`). Label 1 if the smallest p-value among that outcome's
analyses is below 0.05. 5,112 trials, 60% significant.

**Fresh test set.** Defined in writing before any record was fetched. Trials whose results were first posted after our
registry snapshot (CTTI, last results date 2024-10-02):

    AREA[ResultsFirstPostDate]RANGE[2024-10-03,MAX] AND AREA[Phase](PHASE1 OR PHASE2 OR PHASE3)
    AND AREA[InterventionType](DRUG OR BIOLOGICAL) AND AREA[StudyType]INTERVENTIONAL

6,211 trials on 2026-10-30. We dropped trials in any TOP split, in the CTO annotation, or in any set we had used before,
trials that started before 2014 (so none could be in the training pool), and combined phases. 924 have a primary p-value:
Phase II 410, Phase III 481, Phase I 33. Base rate 0.564. Features come from the registration record with the results
section removed. Labels were sealed until the single scoring run.

## 3. Models

Both arms are the same registry model: gradient boosting (scikit-learn `HistGradientBoostingClassifier`, 5 seeds
averaged) on the structured registration fields from `PHASE1-FAILURES.md` section 4, plus one score per text field
(criteria, titles, primary and secondary outcomes, summary, conditions, interventions, arm descriptions), each from a
TF-IDF logistic regression fit out-of-fold on the training rows.

- **TOP label.** Trained on TOP train+valid, plus pre-2014 CTO trials with a soft label (weight 0.25).
- **Endpoint label.** Same features, but the text scores and the boosting model are fit on the 5,112 endpoint labels.

HINT is the repo version trained on TOP's standard split with the upstream settings, 3 seeds averaged, **with the
released (empty) criteria embeddings**. We have not rerun it with the rebuilt file on this set. On TOP test and the CTO
annotation, rebuilding them moved HINT by -0.009 to +0.025 (`PHASE1-FAILURES.md` section 3).

## 4. Results (pre-registered, scored once)

The protocol (arms, metric, sets, the comparisons and the pass rule: paired-bootstrap 95% interval above zero) was
hash-frozen before the fresh labels were read.

| ROC-AUC, 924 fresh trials | Ph II (410) | Ph III (481) | Pooled |
|---|---|---|---|
| registry model, TOP label | 0.625 | 0.690 | 0.688 |
| registry model, endpoint label | **0.690** | **0.709** | **0.723** |
| endpoint minus TOP label | +0.065 [0.017, 0.115] | +0.018 [-0.015, 0.051] | **+0.035 [0.008, 0.060]** |

| ROC-AUC, 277 HINT-featurizable fresh trials | Ph II (110) | Ph III (162) | Pooled |
|---|---|---|---|
| HINT (TOP label, repo as released) | 0.527 | 0.581 | 0.556 |
| registry model, TOP label | 0.547 | 0.645 | 0.644 |
| registry model, endpoint label | **0.652** | **0.694** | **0.715** |
| endpoint label minus HINT | +0.125 [0.022, 0.236] | +0.113 [0.015, 0.197] | **+0.159 [0.095, 0.225]** |

Before the frozen run, the same comparison on an earlier development set of post-2014 trials predicted +0.037 for the
first table. The fresh run gave +0.035.

## 5. Caveats

- **The label is "first reported primary p-value < 0.05", not "the trial succeeded".** It ignores direction,
  non-inferiority margins and which outcome the sponsor considered primary when several are listed. Trials that report no
  p-value (many single-arm trials) are dropped, on both the training and the test side.
- **Phase I is untested.** Only 33 fresh Phase I trials report a primary p-value.
- **The HINT comparison is small (277 trials) and uses the released embeddings.** The rebuilt embeddings are unlikely to
  close a 0.16 gap, given their effect elsewhere, but we have not run it.
- **This is not a TOP benchmark number.** TOP test was not used. HINT was trained for TOP's label and is scored here on a
  different question, so the second table says TOP-trained models transfer poorly to the endpoint question, not that HINT
  is a bad TOP model.

## Reproducing

The fresh set is reproducible from the query above and the public API. Labels: first entry of
`outcomeMeasuresModule.outcomeMeasures` with `type == PRIMARY` that has a parseable `analyses[].pValue`; label 1 if the
smallest of its analysis p-values is below 0.05 (`<0.001` read as just under 0.001). Scripts, predictions and the frozen protocol live outside this repository, in the audit that
produced these numbers.
