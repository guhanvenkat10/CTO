"""Tests for the labeling functions' vote polarity.

Every labeling function returns a vote in Snorkel's two-class encoding: 1 success, 0 failure,
-1 abstain. The distinction that matters, and that these tests pin, is between 0 and -1. A 0 asserts
that the trial failed; a -1 says the channel has no evidence either way. Silently substituting the
first for the second is not a modelling choice with a knob, it is a claim about a trial that the
evidence does not support, and it biases hardest against the most recent cohorts, which are the ones
with the least evidence available.

    pytest labeling/test_lfs.py
"""

import os
import sys

import pandas as pd
import pytest

# Importable when pytest is run from the repository root, not only from this folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lfs import (count_hint_cols, hint_train_lf, lf_column_names, lf_gpt, lf_linkage, lf_pvalues,
                 lf_results_reported, lf_status, split_hint_supervision)

ABSTAIN, FAILURE, SUCCESS = -1, 0, 1


def _csv(tmp_path, name, rows, columns):
    p = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(p, index=False)
    return str(p)


# --------------------------------------------------------------------------- lf_linkage

def test_linkage_recent_trial_abstains(tmp_path):
    """'Recent trial' means the trial has not yet had time to link, not that it failed.

    extract_outcome_from_trial_linkage.py creates this category deliberately, for exactly the trials
    it is not willing to call. Reading it as a failure vote discards that judgement.
    """
    path = _csv(tmp_path, "linkage.csv",
                [("NCT01", "Success"), ("NCT02", "Failure"),
                 ("NCT03", "Not sure"), ("NCT04", "Recent trial")],
                ["nctid", "outcome"])

    df = lf_linkage(path, polarity="abstain").set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == FAILURE
    assert df["NCT03"] == ABSTAIN
    assert df["NCT04"] == ABSTAIN


def test_linkage_default_reproduces_the_release(tmp_path):
    """The default is the released polarity: anything not listed, 'Recent trial' included, votes 0."""
    path = _csv(tmp_path, "linkage.csv",
                [("NCT01", "Success"), ("NCT02", "Failure"),
                 ("NCT03", "Not sure"), ("NCT04", "Recent trial")],
                ["nctid", "outcome"])

    df = lf_linkage(path).set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == FAILURE
    assert df["NCT03"] == ABSTAIN
    assert df["NCT04"] == FAILURE


def test_linkage_rejects_an_unmapped_outcome(tmp_path):
    """A new outcome value should stop the pipeline rather than default to a failure vote."""
    path = _csv(tmp_path, "linkage.csv",
                [("NCT01", "Success"), ("NCT02", "Provisionally linked")],
                ["nctid", "outcome"])

    with pytest.raises(AssertionError, match="unmapped linkage outcome"):
        lf_linkage(path, polarity="abstain")


# --------------------------------------------------------------------------- lf_results_reported

def test_results_reported_abstains_when_nothing_is_posted(tmp_path):
    """Posted results are weak evidence of success; unposted results are no evidence at all.

    Posting is due roughly twelve months after completion and compliance is far from universal, so
    a failure vote here fires on trials that are merely recent or whose sponsor did not post.
    """
    _csv(tmp_path, "calculated_values.txt", [], [])
    p = tmp_path / "calculated_values.txt"
    pd.DataFrame(
        [("NCT01", "t"), ("NCT02", "f"), ("NCT03", "f")],
        columns=["nct_id", "were_results_reported"],
    ).to_csv(p, sep="|", index=False)

    df = lf_results_reported(str(tmp_path) + "/", polarity="abstain").set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == ABSTAIN
    assert df["NCT03"] == ABSTAIN
    assert ABSTAIN in set(df), "this channel must be able to abstain"


def test_results_reported_default_reproduces_the_release(tmp_path):
    """The default is the released polarity, so a fresh clone reproduces old releases exactly."""
    p = tmp_path / "calculated_values.txt"
    pd.DataFrame(
        [("NCT01", "t"), ("NCT02", "f")], columns=["nct_id", "were_results_reported"],
    ).to_csv(p, sep="|", index=False)

    df = lf_results_reported(str(tmp_path) + "/")
    df = df.set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == FAILURE


# --------------------------------------------------------------------------- lf_gpt

def test_gpt_not_sure_abstains(tmp_path):
    """'Not sure' is the model declining to call the outcome.

    In the released pubmed_gpt_outcomes.csv the model returns only 'Success' and 'Not sure' and
    never 'Failure', so under the previous mapping every failure vote this channel cast was the
    model reporting that it could not tell.
    """
    path = _csv(tmp_path, "gpt.csv",
                [("NCT01", "Success"), ("NCT02", "Not sure"), ("NCT03", "Failure")],
                ["nctid", "outcome"])

    df = lf_gpt(path, polarity="abstain").set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == ABSTAIN
    assert df["NCT03"] == FAILURE


def test_gpt_default_reproduces_the_release(tmp_path):
    path = _csv(tmp_path, "gpt.csv",
                [("NCT01", "Success"), ("NCT02", "Not sure"), ("NCT03", "Failure")],
                ["nctid", "outcome"])

    df = lf_gpt(path).set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == FAILURE
    assert df["NCT03"] == FAILURE


@pytest.mark.parametrize("fn", [lf_results_reported, lf_linkage, lf_gpt])
def test_unknown_polarity_is_rejected(fn, tmp_path):
    with pytest.raises(AssertionError, match="polarity must be one of"):
        fn(str(tmp_path) + "/x.csv", polarity="abstian")


# --------------------------------------------------------------------------- hint_train_lf

def test_hint_lf_can_hold_out_the_top_test_split(tmp_path):
    """TOP's held-out test labels should be excludable from the label model.

    In RF/SVM/LR mode this labeling function is also the supervision target, so with the test split
    included those rows help fit the labeler whose output a HINT-family model is later scored
    against. The default keeps them, for reproducibility; this checks the held-out option works.
    """
    for split, nct in [("train", "NCT01"), ("valid", "NCT02"), ("test", "NCT03")]:
        pd.DataFrame([(nct, 1)], columns=["nctid", "label"]).to_csv(
            tmp_path / f"phase1_{split}.csv", index=False)

    legacy = set(hint_train_lf(str(tmp_path))["nct_id"])
    heldout = set(hint_train_lf(str(tmp_path), splits=("train", "valid"))["nct_id"])

    assert legacy == {"NCT01", "NCT02", "NCT03"}
    assert heldout == {"NCT01", "NCT02"}
    assert "NCT03" not in heldout


def test_hint_lf_raises_when_nothing_matches(tmp_path):
    """A silently empty hint labeling function would leave the RF modes with no supervision."""
    with pytest.raises(AssertionError, match="no TOP/HINT files matched"):
        hint_train_lf(str(tmp_path))


# --------------------------------------------------------------------------- lf_status

def test_status_cannot_express_success_for_any_real_trial_status(tmp_path):
    """The status channel is a failure detector with a success branch that never fires.

    Its only success value is 'approved for marketing', which is an expanded-access record type
    rather than a state a completed interventional trial reaches, so its measured success-vote rate
    is 0.000 in every phase and every year 2014-2023. That matters because this channel is also
    applied as a hard override on top of the fitted label model in legacy mode.
    """
    p = tmp_path / "studies.txt"
    pd.DataFrame(
        [("NCT01", "Completed"), ("NCT02", "Terminated"), ("NCT03", "Withdrawn"),
         ("NCT04", "Suspended"), ("NCT05", "Active, not recruiting"), ("NCT06", "Recruiting")],
        columns=["nct_id", "overall_status"],
    ).to_csv(p, sep="|", index=False)

    votes = lf_status(str(tmp_path) + "/").set_index("nct_id")["lf"]

    assert set(votes) <= {ABSTAIN, FAILURE}, "no ordinary trial status can produce a success vote"
    assert votes["NCT02"] == FAILURE
    assert votes["NCT03"] == FAILURE
    assert votes["NCT04"] == FAILURE
    assert votes["NCT01"] == ABSTAIN, "completion alone is not evidence either way"


# --------------------------------------------------------------------------- lf_pvalues

def test_pvalues_is_absent_rather_than_failing_open(tmp_path):
    """A trial with no posted p-value must not appear in this channel's frame at all.

    This is the distinction between lf_pvalues and lf_results_reported, and it is easy to state
    backwards. lf_results_reported reads calculated_values.txt, which is populated for essentially
    every study, so an unposted trial is IN its frame and used to be voted failure. lf_pvalues drops
    rows with no p-value, so an unposted trial is absent and picks up -1 from the fillna in get_lfs.
    """
    pd.DataFrame(
        [(1, "NCT01", 0.01), (2, "NCT02", 0.60), (3, "NCT03", None)],
        columns=["outcome_id", "nct_id", "p_value"],
    ).to_csv(tmp_path / "outcome_analyses.txt", sep="|", index=False)
    pd.DataFrame(
        [(1, "Primary"), (2, "Primary"), (3, "Primary")], columns=["id", "outcome_type"],
    ).to_csv(tmp_path / "outcomes.txt", sep="|", index=False)

    votes = lf_pvalues(str(tmp_path) + "/").set_index("nct_id")["lf"]

    assert votes["NCT01"] == SUCCESS
    assert votes["NCT02"] == FAILURE
    assert "NCT03" not in votes.index, "no posted p-value means absent, not a failure vote"


# --------------------------------------------------------------------------- default preservation

def _fake_lf(name):
    return pd.DataFrame({"nct_id": [f"{name}_1"], "lf": [1]})


def test_legacy_column_layout_is_unchanged_and_dedupe_collapses_it():
    """The default layout is the released 13-column one, in the released order. Asserted on what
    lf_column_names (the helper get_lfs uses) returns, so this guards the code, not a copy of it."""
    legacy = lf_column_names(dedupe_lfs=False)
    dedupe = lf_column_names(dedupe_lfs=True)

    assert legacy == ['hint_train', 'hint_train2', 'hint_train3', 'status', 'status2', 'gpt', 'gpt2',
                      'linkage', 'linkage2', 'stock_price', 'results_reported', 'new_headlines', 'pvalues']
    assert dedupe == ['hint_train', 'status', 'gpt', 'linkage', 'stock_price', 'results_reported',
                      'new_headlines', 'pvalues']
    assert count_hint_cols(legacy) == 3 and count_hint_cols(dedupe) == 1
    # every distinct labeling function survives deduplication
    assert {c.rstrip('23') for c in legacy} == set(dedupe)


@pytest.mark.parametrize("columns, n_hint", [
    (lf_column_names(dedupe_lfs=False), 3),
    (lf_column_names(dedupe_lfs=True), 1),
    ([c for c in lf_column_names(dedupe_lfs=False) if c != 'hint_train'], 2),   # --SKIP_LIST "['hint_train']"
])
def test_hint_column_count_drives_the_feature_slice(columns, n_hint):
    """split_hint_supervision is what the main block uses for RF/SVM/LR. The features must start at the
    first non-hint column, Y must be the first hint column, and rows where hint abstains are dropped."""
    import numpy as np

    columns = list(columns) + ['pred']                       # main appends a pred column before slicing
    rng = np.random.default_rng(0)
    L = rng.integers(-1, 2, size=(50, len(columns)))
    L[:5, 0] = -1
    got_n, X1, Y, X2 = split_hint_supervision(L, columns)

    assert got_n == n_hint
    valid = L[:, 0] != -1
    assert (Y == L[valid, 0]).all()
    assert X1.shape == (valid.sum(), len(columns) - n_hint)
    assert X2.shape == (len(L), len(columns) - n_hint)
    assert not columns[n_hint].startswith('hint_train')       # first feature is never a hint copy
