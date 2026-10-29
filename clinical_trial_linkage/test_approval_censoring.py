"""Tests for the approval-channel censoring flags.

match_FDA_approvals_main needs the cross-encoder, the Orange Book dump and the linkage adjacency
matrices, none of which are in the repository, so the part that can carry a bug is factored out into
two pure functions and tested here.

    pytest clinical_trial_linkage/test_approval_censoring.py
"""

import os
import sys

import pandas as pd
import pytest

# Importable when pytest is run from the repository root, not only from this folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match_fda_approvals import (DEFAULT_APPROVAL_LAG_ALLOWANCE_DAYS, annotate_approval_censoring,
                                 completion_date_map)

HORIZON = pd.Timestamp("2024-04-01")


def _merged(*nctids):
    return pd.DataFrame({"nctid": list(nctids),
                         "outcome": ["Failure"] * len(nctids),
                         "phase": ["phase3"] * len(nctids)})


def test_flags_only_trials_inside_the_lag_allowance():
    """A trial completing close to the snapshot has not had time to produce an approval."""
    df = annotate_approval_censoring(
        _merged("NCT_OLD", "NCT_RECENT"),
        completion_dates={"NCT_OLD": "2015-06-01", "NCT_RECENT": "2023-11-01"},
        orange_book_horizon=HORIZON,
        approval_lag_allowance_days=730,
    ).set_index("nctid")

    assert df.loc["NCT_OLD", "approval_evidence_censored"] is False or \
        not df.loc["NCT_OLD", "approval_evidence_censored"]
    assert df.loc["NCT_RECENT", "approval_evidence_censored"]
    assert df.loc["NCT_OLD", "orange_book_horizon"] == "2024-04-01"
    assert df.loc["NCT_OLD", "approval_lag_allowance_days"] == 730


def test_unknown_completion_date_is_not_flagged_and_does_not_raise():
    """The merged linkage frame carries no completion_date column of its own.

    An earlier version read it from that frame, which silently produced a scalar NaT rather than a
    column and then raised on .fillna. Completion dates come from trial_info instead, and a trial
    missing from it is left unflagged rather than guessed either way.
    """
    df = annotate_approval_censoring(
        _merged("NCT_KNOWN", "NCT_MISSING"),
        completion_dates={"NCT_KNOWN": "2023-11-01"},
        orange_book_horizon=HORIZON,
    ).set_index("nctid")

    assert df.loc["NCT_KNOWN", "approval_evidence_censored"]
    assert not df.loc["NCT_MISSING", "approval_evidence_censored"]
    assert pd.isna(df.loc["NCT_MISSING", "completion_date"])


def test_no_outcome_is_changed():
    """These columns are provenance. They must never move a label."""
    before = _merged("NCT01", "NCT02", "NCT03")
    before.loc[1, "outcome"] = "Success"
    after = annotate_approval_censoring(
        before,
        completion_dates={"NCT01": "2024-01-01", "NCT02": "2016-01-01", "NCT03": "2023-12-01"},
        orange_book_horizon=HORIZON,
    )
    assert list(after["outcome"]) == list(before["outcome"])
    assert list(after["nctid"]) == list(before["nctid"])
    assert after["approval_evidence_censored"].sum() == 2


def test_the_boundary_is_the_allowance_not_the_horizon():
    """A trial exactly on the cutoff is not flagged; one day later is."""
    cutoff = HORIZON - pd.Timedelta(days=DEFAULT_APPROVAL_LAG_ALLOWANCE_DAYS)
    df = annotate_approval_censoring(
        _merged("NCT_ON", "NCT_AFTER"),
        completion_dates={"NCT_ON": cutoff.strftime("%Y-%m-%d"),
                          "NCT_AFTER": (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")},
        orange_book_horizon=HORIZON,
    ).set_index("nctid")

    assert not df.loc["NCT_ON", "approval_evidence_censored"]
    assert df.loc["NCT_AFTER", "approval_evidence_censored"]


def test_completion_dates_are_read_from_nested_trial_info():
    """trial_info nests some fields one level deep, which is how the matching code reads it too."""
    trial_info = {
        "NCT_FLAT": {"completion_date": "2020-01-01"},
        "NCT_NESTED": {"dates": {"completion_date": "2021-01-01"}},
        "NCT_NONE": {"phase": "phase3"},
    }
    got = completion_date_map(trial_info)
    assert got == {"NCT_FLAT": "2020-01-01", "NCT_NESTED": "2021-01-01"}
