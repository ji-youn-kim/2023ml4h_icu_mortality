import numpy as np

def map_dict(cohort, colname):
    
    code2id = {"pad": 0, "unk": 1}
    for idx, i in enumerate(cohort[colname].unique()):
        code2id[i] = idx + 2
    
    return code2id


def scale_numeric(val, minmax):

    x_std = (val - minmax[0]) / (minmax[1] - minmax[0])
    x_scaled = x_std * (1 - (-1)) + (-1)

    return x_scaled


def update_dict(cohort_keys, cur_icu, itemid2dict, time_minmax, val_minmax):

    # Sort CHARTEVENTS by time
    events = sorted(cur_icu["CHARTEVENTS"])
    
    update_target = dict()
    for key in cohort_keys:
        update_target[key] = cur_icu[key]
    update_target["CHARTEVENTS"] = []
    for event in events:
        update_target["CHARTEVENTS"].append((scale_numeric(event[0], time_minmax), map_value(event[1], itemid2dict), scale_numeric(event[2], val_minmax)))
    
    return update_target


def map_value(x, code2id):
    if x in code2id.keys():
        return code2id[x]
    else:
        return code2id["unk"]
        

def convert(cohort, code2id, colname):

    cohort = cohort.copy()
    cohort[colname] = cohort[colname].apply(lambda x: map_value(x, code2id))

    return cohort


def icu_itemids(icu_dict, x_samples, y_samples, itemid2dict):
    
    for idx, key in enumerate(icu_dict):
        item_dict = dict()
        icustay = np.zeros(len(itemid2dict))
        for event in icu_dict[key]["CHARTEVENTS"]:
            if event[1] not in item_dict:
                item_dict[event[1]] = [event[2], 1]
            else:
                item_dict[event[1]][0] += event[2]
                item_dict[event[1]][1] += 1
        for it_key in item_dict:
            icustay[it_key] = item_dict[it_key][0] / item_dict[it_key][1]
        
        x_samples[idx] = icustay
        y_samples[idx] = icu_dict[key]["LABEL"]

    return x_samples, y_samples