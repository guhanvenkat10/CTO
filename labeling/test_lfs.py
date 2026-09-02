"""Tests for the labeling functions' vote polarity.

Every labeling function returns a vote in Snorkel's two-class encoding: 1 success, 0 failure,
-1 abstain. The distinction that matters, and that these tests pin, is between 0 and -1. A 0 asserts
that the trial failed; a -1 says the channel has no evidence either way. Silently substituting the
first for the second is not a modelling choice with a knob, it is a claim about a trial that the
evidence does not support, and it biases hardest against the most recent cohorts, which are the ones
with the least evidence available.

    pytest labeling/test_lfs.py
"""

import pandas as pd
import pytest

from lfs import lf_gpt, lf_linkage, lf_results_reported

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

    df = lf_linkage(path).set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == FAILURE
    assert df["NCT03"] == ABSTAIN
    assert df["NCT04"] == ABSTAIN


def test_linkage_rejects_an_unmapped_outcome(tmp_path):
    """A new outcome value should stop the pipeline rather than default to a failure vote."""
    path = _csv(tmp_path, "linkage.csv",
                [("NCT01", "Success"), ("NCT02", "Provisionally linked")],
                ["nctid", "outcome"])

    with pytest.raises(AssertionError, match="unmapped linkage outcome"):
        lf_linkage(path)


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

    df = lf_results_reported(str(tmp_path) + "/").set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == ABSTAIN
    assert df["NCT03"] == ABSTAIN
    assert ABSTAIN in set(df), "this channel must be able to abstain"


def test_results_reported_legacy_flag_restores_the_failure_vote(tmp_path):
    """The previous behaviour stays reachable, so old releases can be reproduced exactly."""
    p = tmp_path / "calculated_values.txt"
    pd.DataFrame(
        [("NCT01", "t"), ("NCT02", "f")], columns=["nct_id", "were_results_reported"],
    ).to_csv(p, sep="|", index=False)

    df = lf_results_reported(str(tmp_path) + "/", abstain_when_unposted=False)
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

    df = lf_gpt(path).set_index("nct_id")["lf"]

    assert df["NCT01"] == SUCCESS
    assert df["NCT02"] == ABSTAIN
    assert df["NCT03"] == FAILURE
