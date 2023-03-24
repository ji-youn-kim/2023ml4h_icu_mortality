import os
import sys
import logging
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import numpy as np
from preprocess.utils import map_dict, update_dict, convert, icu_itemids

logger = logging.getLogger(__name__)


def build_cohort(args):

    logger.info("Start Building Cohort...")
    
    columns = {"icustays": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"], \
                "admissions": ["HADM_ID", "DEATHTIME", "ADMISSION_TYPE", "ADMISSION_LOCATION", "DIAGNOSIS"]}
    times = {"icustays": ["INTIME", "OUTTIME"], "admissions": ["DEATHTIME"]}
    map_columns = ["ADMISSION_TYPE", "ADMISSION_LOCATION", "DIAGNOSIS"]
    
    # 1. Filter ICUSTAYS by Length-of-Stay
    icustays = pd.read_csv(os.path.join(args.source_path, "ICUSTAYS.csv"))
    for t in times["icustays"]:
        icustays[t] = pd.to_datetime(icustays[t])
    cohort = icustays.loc[(args.min_los <= icustays["LOS"]) & (icustays["LOS"] <= args.max_los)]
    cohort = cohort[columns["icustays"]].reset_index(drop=True)

    # 2. Merge ADMISSIONS with features; admission type, admission location, diagnosis, deathtime
    admissions = pd.read_csv(os.path.join(args.source_path, "ADMISSIONS.csv"))
    for t in times["admissions"]:
        admissions[t] = pd.to_datetime(admissions[t])
    cohort = cohort.merge(admissions[columns["admissions"]], how="inner", on="HADM_ID")
    cohort = cohort.reset_index(drop=True)

    # 3. Create Death Label: ICU Intime <= ADMISSIONS Deathtime <= ICU Outtime
    cohort["LABEL"] = 0
    cohort.loc[((cohort["INTIME"] <= cohort["DEATHTIME"]) & (cohort["DEATHTIME"] <= cohort["OUTTIME"])), "LABEL"] = 1
    
    train_cohort = cohort.loc[~(cohort["ICUSTAY_ID"] % 10 == 8) & ~(cohort["ICUSTAY_ID"] % 10 == 9)]
    test_cohort = cohort.loc[(cohort["ICUSTAY_ID"] % 10 == 8) | (cohort["ICUSTAY_ID"] % 10 == 9)]

    # Create ID Dictionary for Categorical Columns
    code2id = dict()
    for m_c in map_columns:
        code2id[m_c] = map_dict(train_cohort, m_c)
    
    # Modify Categorical Column Values to ID
    for m_c in map_columns:
        train_cohort = convert(train_cohort, code2id[m_c], m_c)
        test_cohort = convert(test_cohort, code2id[m_c], m_c)

    train_cohort.to_csv(os.path.join(args.input_path, "train_cohort.csv"))
    test_cohort.to_csv(os.path.join(args.input_path, "test_cohort.csv"))

    for key in code2id.keys():
        with open(os.path.join(args.input_path, f"{key.lower()}2id.pkl"), "wb") as f:
            pkl.dump(code2id[key], f, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info(f"Done Creating Cohort for {len(train_cohort)} Train | {len(test_cohort)} Test ICUSTAYS")

    return train_cohort, test_cohort


# For Processing RNN Inputs
def process_events(args, train_cohort, test_cohort):

    logger.info("Start Processing CHARTEVENTS...")

    icu_dict, train_icu_dict, test_icu_dict = dict(), dict(), dict()
    itemid2dict = {'pad': 0, 'unk': 1}
    train_time_minmax, test_time_minmax, train_val_minmax, test_val_minmax = [np.array([sys.maxsize, 0]) for _ in range(4)]
    cohort = pd.concat([train_cohort, test_cohort])
    cohort_ids = cohort.ICUSTAY_ID.unique()
    times = {"chartevents": ["CHARTTIME"]}
    cohort_keys = ["LABEL", "ADMISSION_TYPE", "ADMISSION_LOCATION", "DIAGNOSIS"]

    with tqdm(total = 330712483, desc = "Processing: ") as pbar:
        for chunk in pd.read_csv(os.path.join(args.source_path, "CHARTEVENTS.csv"), chunksize=args.chunksize):
            nas = chunk["ICUSTAY_ID"].isna().sum() + chunk["VALUENUM"].isna().sum()
            chunk.dropna(subset=["ICUSTAY_ID", "VALUENUM"], inplace=True)
            pbar.update(nas)
            chunk["ICUSTAY_ID"] = chunk["ICUSTAY_ID"].astype(int)
            for t in times["chartevents"]:
                chunk[t] = pd.to_datetime(chunk[t])
            
            for _, row in chunk.iterrows():
                # Check if ICUSTAY ID is in cohort
                if row["ICUSTAY_ID"] not in cohort_ids:
                    pbar.update(1)
                    continue
                
                # Check if CHARTTIME is within first 3 hrs of INTIME
                delta = (row["CHARTTIME"] - cohort[cohort["ICUSTAY_ID"] == row["ICUSTAY_ID"]].iloc[0]["INTIME"]).total_seconds() / 60
                if not (0 <= delta and delta <= args.time_limit * 60):
                    pbar.update(1)
                    continue
                
                # Create a ICUSTAY ID dictionary
                if row["ICUSTAY_ID"] not in icu_dict.keys():
                    icu_dict[row["ICUSTAY_ID"]] = dict()
                    for key in cohort_keys:
                        icu_dict[row["ICUSTAY_ID"]][key] = cohort[cohort["ICUSTAY_ID"] == row["ICUSTAY_ID"]].iloc[0][key]
                    icu_dict[row["ICUSTAY_ID"]]["CHARTEVENTS"] = []
                # Limit # of CHARTEVENTS to 300 per ICUSTAY ID
                elif len(icu_dict[row["ICUSTAY_ID"]]["CHARTEVENTS"]) >= args.max_event_cnt:
                    pbar.update(1)
                    continue
                
                icu_dict[row["ICUSTAY_ID"]]["CHARTEVENTS"].append((delta, row["ITEMID"], row["VALUENUM"]))
                 
                # Save Min-Max of Timegap, Value for MinMax Scaling 
                if row["ICUSTAY_ID"] % 10 not in [8,9]:
                    if row["ITEMID"] not in itemid2dict:
                        itemid2dict[row["ITEMID"]] = len(itemid2dict)
                    if delta < train_time_minmax[0]:
                        train_time_minmax[0] = delta
                    if delta > train_time_minmax[1]:
                        train_time_minmax[1] = delta
                    if row["VALUENUM"] < train_val_minmax[0]:
                        train_val_minmax[0] = row["VALUENUM"]
                    if row["VALUENUM"] > train_val_minmax[1]:
                        train_val_minmax[1] = row["VALUENUM"]
                else:
                    if delta < test_time_minmax[0]:
                        test_time_minmax[0] = delta
                    if delta > test_time_minmax[1]:
                        test_time_minmax[1] = delta
                    if row["VALUENUM"] < test_val_minmax[0]:
                        test_val_minmax[0] = row["VALUENUM"]
                    if row["VALUENUM"] > test_val_minmax[1]:
                        test_val_minmax[1] = row["VALUENUM"]
                
                pbar.update(1)
    
    # 1) Split Train / Test into Separate Dictionaries, and 2) Apply Min-Max Scaling for Numeric Columns
    for icustay in icu_dict:
        if icustay % 10 not in [8,9]:
            train_icu_dict[icustay] = update_dict(cohort_keys, icu_dict[icustay], itemid2dict, train_time_minmax, train_val_minmax)
        else:
            test_icu_dict[icustay] = update_dict(cohort_keys, icu_dict[icustay], itemid2dict, test_time_minmax, test_val_minmax) 

    with open(os.path.join(args.input_path, "itemid2dict.pkl"), "wb") as f:
        pkl.dump(itemid2dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(args.input_path, "X_train_rnn.pkl"), "wb") as f:
        pkl.dump(train_icu_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(args.input_path, "X_test_rnn.pkl"), "wb") as f:
        pkl.dump(test_icu_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    logger.info(f"Done Processing Events for {len(train_icu_dict)} Train | {len(test_icu_dict)} Test ICUSTAYS")

    return train_icu_dict, test_icu_dict


# For Processing Logistic Regression Inputs
def batch_itemids(args, train_icu_dict, test_icu_dict):
    
    itemid2dict = pd.read_pickle(os.path.join(args.input_path, "itemid2dict.pkl"))
    x_train_samples = np.zeros((len(train_icu_dict), len(itemid2dict)))
    x_test_samples = np.zeros((len(test_icu_dict), len(itemid2dict)))
    y_train_samples = np.zeros((len(train_icu_dict)))
    y_test_samples = np.zeros((len(test_icu_dict)))

    x_train_samples, y_train_samples = icu_itemids(train_icu_dict, x_train_samples, y_train_samples, itemid2dict)
    x_test_samples, y_test_samples = icu_itemids(test_icu_dict, x_test_samples, y_test_samples, itemid2dict)

    with open(os.path.join(args.input_path, 'X_train_logistic.npy'), 'wb') as f:
        np.save(f, x_train_samples)
    with open(os.path.join(args.input_path, 'X_test_logistic.npy'), 'wb') as f:
        np.save(f, x_test_samples)
    with open(os.path.join(args.input_path, 'y_train.npy'), 'wb') as f:
        np.save(f, y_train_samples)
    with open(os.path.join(args.input_path, 'y_test.npy'), 'wb') as f:
        np.save(f, y_test_samples)

    logger.info(f"Done Creating {x_train_samples.shape} Train | {x_test_samples.shape} Test ICU ITEMID Numpy Arrays")

    return x_train_samples, x_test_samples