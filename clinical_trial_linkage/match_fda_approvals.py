from ast import arg
from re import match
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import timedelta
import argparse
# sentence_transformers is imported at the entry point rather than here. The cross-encoder is only
# ever constructed in __main__ and passed in as an argument, and importing it at module scope pulls
# in torch, which means nothing in this file can be imported or tested without a full GPU stack.

import warnings
warnings.filterwarnings('ignore')


def get_sub_search_group_FDA_approval(approval_dict, phase_3_trials_df):
        fda_approval_date = approval_dict['Approval_Date']
        
        # get all the trial with completion date before the approval date in the phase_3_trials_df
        phase_3_trials_df['completion_date'] = pd.to_datetime(phase_3_trials_df['completion_date'])
        phase_3_trials_df['completion_date'] = phase_3_trials_df['completion_date'].dt.strftime('%Y-%m-%d')
        # similar_trials = phase_3_trials_df[phase_3_trials_df['completion_date'] < fda_approval_date]
        similar_trials = phase_3_trials_df[phase_3_trials_df['completion_date'] < (datetime.strptime(fda_approval_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')]
        # remove the trials with completion date 2 years before the approval date
        similar_trials = similar_trials[similar_trials['completion_date'] > (datetime.strptime(fda_approval_date, '%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')]
                

        return similar_trials

def match_FDA_approval_to_trials(approval_dict, similar_trials,cross_encoder):
    ingredient = approval_dict['Ingredient'].lower()
    applicant  = approval_dict['Applicant'].lower()
    trade_name = approval_dict['Trade_Name'].lower()
    fda_approval_date = pd.to_datetime(approval_dict['Approval_Date'], format="%Y-%m-%d")
    
    # Ensure each element in the columns is a string
    similar_trials['generic_name'] = similar_trials['generic_name'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else x.lower())
    similar_trials['intervention_name'] = similar_trials['intervention_name'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else x.lower())

    # Filter using ingredient in intervention generic name
    matched_trials = similar_trials[similar_trials['generic_name'].apply(lambda x: ingredient in x)]
    
    # Initialize the 'generic_name_score' column
    matched_trials['generic_name_score'] = 0.0
    # Get scores from cross encoder comparing generic name with ingredient
    for i in range(matched_trials.shape[0]):
        matched_trials.at[matched_trials.index[i], 'generic_name_score'] = cross_encoder.predict([(ingredient, matched_trials.iloc[i]['generic_name'])])[0]
        
    # Get top 5 trials with highest scores
    matched_trials = matched_trials.sort_values(by='generic_name_score', ascending=False).head(5)
    

    if matched_trials.empty:
        return ''
    
    # Convert completion dates once
    matched_trials['completion_date'] = pd.to_datetime(matched_trials['completion_date'], format="%Y-%m-%d", errors='coerce')
    # Calculate the difference in days directly
    matched_trials['days_diff'] = (fda_approval_date - matched_trials['completion_date']).dt.days
    
    # Filter out rows where days_diff is negative
    matched_trials = matched_trials[matched_trials['days_diff'] >= 0]
    
    if matched_trials.empty:
        return ''
    
    # Get the trial with the smallest non-negative days_diff
    matched_trials_applicant = matched_trials.loc[matched_trials['days_diff'].idxmin(), 'nctid']
    
    return matched_trials_applicant


# Days a Phase 3 trial is allowed, after primary completion, for a first-ever Orange Book entry
# to plausibly appear. Used only to flag censored rows, never to change a label.
#
# Measured the same way as the linkage windows: Phase 3 and Phase 2/3 trials with an actual primary
# completion in 2014-2016 (a decade of follow-up, so observed lags are close to true ones), an
# intervention whose first-ever Orange Book entry comes after completion, each first approval matched
# to the closest preceding completion as in match_FDA_approvals_main. n = 55 matched first approvals:
# lag median 534 days, 25th percentile 255, 75th 935, 90th 1,566. (2010-2016, n = 118: median 447.)
# The default is the median, as for the linkage windows. The previous 730 covered 69% of matched
# approvals. Pass --approval_lag_allowance_days 935 (75th) or 1566 (90th) for a stricter flag.
DEFAULT_APPROVAL_LAG_ALLOWANCE_DAYS = 534


def completion_date_map(trial_info):
    """{nctid: completion_date} from the trial_info dict, which nests fields one level deep."""
    out = {}
    for nctid, rec in trial_info.items():
        if not isinstance(rec, dict):
            continue
        value = rec.get('completion_date')
        if value is None:
            for sub in rec.values():
                if isinstance(sub, dict) and sub.get('completion_date') is not None:
                    value = sub['completion_date']
                    break
        if value is not None:
            out[nctid] = value
    return out


def annotate_approval_censoring(merged_all_pd, completion_dates, orange_book_horizon,
                                approval_lag_allowance_days=DEFAULT_APPROVAL_LAG_ALLOWANCE_DAYS):
    """Flag the rows for which absence of an approval is uninformative, and record why.

    Adds four columns (completion_date, orange_book_horizon, approval_lag_allowance_days,
    approval_evidence_censored) and changes no outcome. A trial completing within
    ``approval_lag_allowance_days`` of the snapshot horizon has not had time to produce a first-ever
    Orange Book entry, so "no approval found" cannot be distinguished from "no approval could have
    been found yet". A trial whose completion date is unknown is left unflagged rather than guessed.

    Kept separate from match_FDA_approvals_main so it can be tested without the cross-encoder, the
    Orange Book dump or the linkage adjacency matrices -- none of which are in the repository.
    """
    df = merged_all_pd.copy()
    horizon = pd.Timestamp(orange_book_horizon)
    cutoff = horizon - pd.Timedelta(days=int(approval_lag_allowance_days))

    completion = pd.to_datetime(df['nctid'].map(completion_dates), errors='coerce')
    df['completion_date'] = completion.dt.strftime('%Y-%m-%d')
    df['orange_book_horizon'] = horizon.strftime('%Y-%m-%d')
    df['approval_lag_allowance_days'] = int(approval_lag_allowance_days)
    df['approval_evidence_censored'] = (completion > cutoff).fillna(False).astype(bool)
    return df


def match_FDA_approvals_main(save_path,merged_all_pd_path,cross_encoder,dev=False,
                             approval_lag_allowance_days=DEFAULT_APPROVAL_LAG_ALLOWANCE_DAYS):
    print('Reading the FDA approvals')
    # read the FDA approvals
    FDA_path = save_path.replace('clinical_trial_linkage/trial_linkages','FDA/FDA_new/products.txt')
    orange_book = pd.read_csv(FDA_path, sep='~',parse_dates=['Approval_Date'])
    # convert 'Approved Prior to Jan 1, 1982' to Jan 1, 1982
    orange_book['Approval_Date'] = orange_book['Approval_Date'].replace('Approved Prior to Jan 1, 1982', '1982-01-01')
    orange_book['Approval_Date'] = pd.to_datetime(orange_book['Approval_Date'])
    orange_book['Approval_Date'] = orange_book['Approval_Date'].dt.strftime('%Y-%m-%d')
    # extract only the following columns Ingredient, Trade_Name, Applicant and Approval_Date
    orange_book = orange_book[['Ingredient', 'Trade_Name', 'Applicant', 'Approval_Date']]
    # remove duplicates
    orange_book = orange_book.drop_duplicates()
    orange_book.reset_index(drop=True, inplace=True)
    print(f'Number of unique FDA approvals: {orange_book.shape[0]}')
    
    #  read trial info 
    trial_info_path = save_path.replace('trial_linkages','trial_info.json')
    with open(trial_info_path, 'r') as f:
        trial_info = json.load(f)
        f.close()
    trial_info[list(trial_info.keys())[0]]

    #extract phase 3 and phase 2/ phase 3 trials with drug intervention type
    phase_3_trials = {}
    for study in trial_info:
        if trial_info[study]['phase'] == 'phase3' or trial_info[study]['phase'] == 'phase2/phase3':
            if 'DRUG' in trial_info[study]['interventions']['intervention_type']:
                phase_3_trials[study] = trial_info[study]
    
    print(len(phase_3_trials))
    # expand the nested dictionary to a flat dictionary
    phase_3_trials_flat = {}    
    for study in phase_3_trials:
        phase_3_trials_flat[study] = {}
        for key in phase_3_trials[study]:
            if isinstance(phase_3_trials[study][key], dict):
                for k in phase_3_trials[study][key]:
                    phase_3_trials_flat[study][k] = phase_3_trials[study][key][k]
            else:
                phase_3_trials_flat[study][key] = phase_3_trials[study][key]  
                
    # convert phase_3_trials to a dataframe
    phase_3_trials_df = pd.DataFrame.from_dict(phase_3_trials_flat, orient='index')
    #rename the index to nctid
    phase_3_trials_df.reset_index(inplace=True)
    phase_3_trials_df.rename(columns={'index':'nctid'}, inplace=True)

    print(phase_3_trials_df.columns)
    phase_3_trials_df['generic_name'] = phase_3_trials_df['generic_name'].apply(lambda x: list(OrderedDict.fromkeys(x)))
            
             
                
    phase_3_fda_matched = {}

    for i in tqdm(range(orange_book.shape[0])):
        approval_dict = orange_book.loc[i]
        similar_trials = get_sub_search_group_FDA_approval(approval_dict, phase_3_trials_df)
        matched_trial = match_FDA_approval_to_trials(approval_dict, similar_trials,cross_encoder)
        # print(matched_trial)
        # print(matched_trial)
        if matched_trial != '':
            phase_3_fda_matched[matched_trial] = {}
            phase_3_fda_matched[matched_trial]['Approval_Date'] = approval_dict['Approval_Date']
            phase_3_fda_matched[matched_trial]['Ingredient'] = approval_dict['Ingredient']
            phase_3_fda_matched[matched_trial]['Trade_Name'] = approval_dict['Trade_Name']
            phase_3_fda_matched[matched_trial]['Applicant'] = approval_dict['Applicant']
            # add following information intervention name, intervention generic name, start_date, completion date condition and lead sponsor
            trial_phase_3_info = phase_3_trials_df[phase_3_trials_df['nctid'] == matched_trial]
            phase_3_fda_matched[matched_trial]['intervention_name'] = trial_phase_3_info['intervention_name'].values[0]
            phase_3_fda_matched[matched_trial]['generic_name'] = trial_phase_3_info['generic_name'].values[0]
            phase_3_fda_matched[matched_trial]['start_date'] = trial_phase_3_info['start_date'].values[0]
            phase_3_fda_matched[matched_trial]['completion_date'] = trial_phase_3_info['completion_date'].values[0]
            phase_3_fda_matched[matched_trial]['condition'] = trial_phase_3_info['conditions'].values[0]
            phase_3_fda_matched[matched_trial]['lead_sponsor'] = trial_phase_3_info['lead_sponsor'].values[0]
        if dev and i == 20:
            print('Development break after 20 iterations')
            break
    print(f'Number of matched trials: {len(phase_3_fda_matched)}')
    
    
    # convert to dataframe
    phase_3_fda_matched_df = pd.DataFrame.from_dict(phase_3_fda_matched, orient='index')
    phase_3_fda_matched_df.reset_index(inplace=True)
    phase_3_fda_matched_df.rename(columns={'index':'nctid'},inplace=True)
    phase_3_fda_matched_df.to_csv(os.path.join(save_path, 'phase_3_fda_matched.csv'), index=False)
    
    
    print('Finished matching FDA approvals to trials')

    
    # read the merged_all_pd
    merged_all_pd = pd.read_csv(merged_all_pd_path)
    
    # check if nctid is in phase_3_fda_matched
    # if true and outcome in merged_all_pd is not Success convert to Success
    # else leave it
    
    matched_nctids = set(phase_3_fda_matched_df['nctid'].values)
    for i in range(merged_all_pd.shape[0]):
        nctid = merged_all_pd.iloc[i]['nctid']
        if nctid in matched_nctids:
            if merged_all_pd.iloc[i]['outcome'] != 'Success':
                merged_all_pd.at[i, 'outcome'] = 'Success'

    # This channel is the one that actually assigns Phase 3 success, and it can only ever add
    # successes -- there is no branch that adds a failure. So the horizon of the Orange Book
    # snapshot is a hard bound on how much success evidence a cohort can show, and the bound bites
    # hardest on the most recent trials, which have had the least time to produce an approval.
    #
    # `approval_evidence_censored` marks the trials for which "no approval found" cannot be
    # distinguished from "no approval could have been found yet". It changes no label; it makes the
    # limit legible to anyone reading the released file, which currently records neither the
    # snapshot date nor the lag allowance.
    #
    # Two further limits that no flag can fix, and that belong next to any use of this channel:
    # the Orange Book lists small molecules only, so every biologic is invisible to it, and a
    # first-ever entry is required, so approval of an already-marketed drug for a new indication
    # never registers here.
    # The merged frame carries only nctid/outcome/phase/linkage columns, so completion dates come
    # from trial_info, which is the same source the matching above uses.
    orange_book_horizon = pd.to_datetime(orange_book['Approval_Date']).max()
    merged_all_pd = annotate_approval_censoring(
        merged_all_pd,
        completion_dates=completion_date_map(trial_info),
        orange_book_horizon=orange_book_horizon,
        approval_lag_allowance_days=approval_lag_allowance_days,
    )
    n_censored = int(merged_all_pd['approval_evidence_censored'].sum())
    n_unknown = int(merged_all_pd['completion_date'].isna().sum())
    print(f'Orange Book horizon {orange_book_horizon:%Y-%m-%d}; '
          f'{n_censored} trials completed within {approval_lag_allowance_days} d of it and are '
          f'flagged approval_evidence_censored (their labels are unchanged); '
          f'{n_unknown} trials have no completion date and are not flagged')

    # merge_all_save_path = merged_all_pd_path.split('.csv')[0] + '_FDA_updated.csv'
    merge_all_save_path = os.path.join(save_path, 'outcome_labels','Merged_all_trial_linkage_outcome_df__FDA_updated.csv' )
    merged_all_pd.to_csv(merge_all_save_path, index=False)
    
    print('Finished updating merged_all_pd with FDA approvals')
    
if __name__ == '__main__':
    from sentence_transformers import CrossEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='', help='path to save the data')
    # parser.add_argument('--trial_linkage_path', type=str, default = None, help='Path to save the matched trials results (provide the path where the trial linkages results are saved)')
    parser.add_argument('--dev', action='store_true', help='Development mode')
    parser.add_argument('--approval_lag_allowance_days', type=int,
                        default=DEFAULT_APPROVAL_LAG_ALLOWANCE_DAYS,
                        help='Days after primary completion within which a first-ever Orange '
                             'Book entry could plausibly appear. Trials completing within this '
                             'many days of the snapshot horizon are flagged '
                             'approval_evidence_censored. Flag only; no label changes.')
    args = parser.parse_args()
    
    trial_linkage_path = os.path.join(args.save_path, 'clinical_trial_linkage/trial_linkages')
    
    save_path = trial_linkage_path # Path to save the matched trials results (provide the path where the trial linkages results are saved)
    if save_path is None:
        raise ValueError('Please provide the path to save the matched trials results (provide the path where the trial linkages results are saved)')
    merged_all_pd_path = os.path.join(save_path, 'outcome_labels','Merged_(ALL)_trial_linkage_outcome_df.csv')
    device = 'cuda'
    cross_encoder = CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)

    
    print(f'Matching FDA approvals to trials and updating the merged_all_pd at {merged_all_pd_path}')
    match_FDA_approvals_main(save_path,merged_all_pd_path,cross_encoder,dev=args.dev,
                             approval_lag_allowance_days=args.approval_lag_allowance_days)
    
# 4