"""Tests for the linkage censoring window.

The window must be the window of the trial being labeled, not of the next phase it would link to.
link_outcome is the per-trial decision factored out of get_trial_linkage_weak_outcome_labels, so it can
be tested without the linkage files.

    pytest clinical_trial_linkage/test_linkage_censoring.py
"""

import os
import sys

import pandas as pd
import pytest

# Importable when pytest is run from the repository root, not only from this folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_outcome_from_trial_linkage import (DEFAULT_CENSORING_WINDOW_DAYS, LEGACY_CENSORING_WINDOW_DAYS,
                                                link_outcome)

AS_OF = pd.Timestamp("2024-01-01")
DONE_500_DAYS_AGO = AS_OF - pd.Timedelta(days=500)


def test_phase1_unlinked_500_days_is_failure():
    # Phase 1 window is 469 days, so a trial completed 500 days ago has had its chance to link.
    outcome, window = link_outcome(False, False, DONE_500_DAYS_AGO, "phase1", AS_OF)
    assert window == DEFAULT_CENSORING_WINDOW_DAYS["phase1"] == 469
    assert outcome == "Failure"


def test_phase2_unlinked_500_days_is_recent():
    # Phase 2 window is 648 days, so the same dates are still inside the window.
    outcome, window = link_outcome(False, False, DONE_500_DAYS_AGO, "phase2", AS_OF)
    assert window == DEFAULT_CENSORING_WINDOW_DAYS["phase2"] == 648
    assert outcome == "Recent trial"


@pytest.mark.parametrize("phase", sorted(DEFAULT_CENSORING_WINDOW_DAYS))
def test_every_phase_uses_its_own_window(phase):
    _, window = link_outcome(False, False, DONE_500_DAYS_AGO, phase, AS_OF)
    assert window == DEFAULT_CENSORING_WINDOW_DAYS[phase]


def test_override_applies_to_every_phase_and_183_reproduces_legacy():
    for phase in DEFAULT_CENSORING_WINDOW_DAYS:
        _, window = link_outcome(False, False, DONE_500_DAYS_AGO, phase, AS_OF, override=LEGACY_CENSORING_WINDOW_DAYS)
        assert window == 183


def test_link_states():
    old = AS_OF - pd.Timedelta(days=2000)
    assert link_outcome(True, True, old, "phase2", AS_OF)[0] == "Success"
    assert link_outcome(True, False, old, "phase2", AS_OF)[0] == "Not sure"
    assert link_outcome(False, False, old, "phase2", AS_OF)[0] == "Failure"
    recent = AS_OF - pd.Timedelta(days=10)
    assert link_outcome(True, True, recent, "phase2", AS_OF)[0] == "Success"     # a positive link wins even if recent
    assert link_outcome(True, False, recent, "phase2", AS_OF)[0] == "Recent trial"
