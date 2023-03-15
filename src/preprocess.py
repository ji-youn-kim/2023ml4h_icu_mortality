import os
import logging
import pandas as pd
import pickle as pkl
from tqdm import tqdm

logger = logging.getLogger(__name__)

def build_cohort(args):

    logger.info("Start Building Cohort...")
    
    columns = {"icustays": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"], \
                "admissions": ["HADM_ID", "DEATHTIME", "ADMISSION_TYPE", "ADMISSION_LOCATION", "DIAGNOSIS"]}
    times = {"icustays": ["INTIME", "OUTTIME"], "admissions": ["DEATHTIME"]}
    
    # 1. Filter ICUSTAYS by Length-of-Stay
    icustays = pd.read_csv(os.path.join(args.input_path, "ICUSTAYS.csv"))
    for t in times["icustays"]:
        icustays[t] = pd.to_datetime(icustays[t])
    cohort = icustays.loc[(args.min_los <= icustays["LOS"]) & (icustays["LOS"] <= args.max_los)]
    cohort = cohort[columns["icustays"]].reset_index(drop=True)

    # 2. Merge ADMISSIONS with features; admission type, admission location, diagnosis, deathtime
    admissions = pd.read_csv(os.path.join(args.input_path, "ADMISSIONS.csv"))
    for t in times["admissions"]:
        admissions[t] = pd.to_datetime(admissions[t])
    cohort = cohort.merge(admissions[columns["admissions"]], how="inner", on="HADM_ID")
    cohort = cohort.reset_index(drop=True)

    # 3. Create Death Label: ICU Intime <= ADMISSIONS Deathtime <= ICU Outtime
    cohort["LABEL"] = 0
    cohort.loc[((cohort["INTIME"] <= cohort["DEATHTIME"]) & (cohort["DEATHTIME"] <= cohort["OUTTIME"])), "LABEL"] = 1
    
    cohort.to_csv(os.path.join(args.output_path, "cohort.csv"))
    logger.info(f"Done Creating Cohort for {len(cohort)} ICUSTAYS")

    return cohort


def process_events(args, cohort):

    logger.info("Start Processing CHARTEVENTS...")

    icu_dict = dict()
    cohort_ids = cohort.ICUSTAY_ID.unique()
    times = {"chartevents": ["CHARTTIME"]}
    cohort_keys = ["LABEL", "ADMISSION_TYPE", "ADMISSION_LOCATION", "DIAGNOSIS"]

    with tqdm(total = 330712483, desc = "Processing: ") as pbar:
        for chunk in pd.read_csv(os.path.join(args.input_path, "CHARTEVENTS.csv"), chunksize=args.chunksize):
            nas = chunk["ICUSTAY_ID"].isna().sum() + chunk["VALUENUM"].isna().sum()
            chunk.dropna(subset=["ICUSTAY_ID", "VALUENUM"], inplace=True)
            pbar.update(nas)
            chunk["ICUSTAY_ID"] = chunk["ICUSTAY_ID"].astype(int)
            for t in times["chartevents"]:
                chunk[t] = pd.to_datetime(chunk[t])
            
            for index, row in chunk.iterrows():
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
                pbar.update(1)
    
    with open(os.path.join(args.output_path, "icustays.pkl"), "wb") as f:
        pkl.dump(icu_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    logger.info(f"Done Processing Events for {len(cohort)} ICUSTAYS")

    return cohort, icu_dict