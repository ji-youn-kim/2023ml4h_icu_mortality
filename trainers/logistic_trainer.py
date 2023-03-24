import os
import logging
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

def LRTrainer(args):
    
    x_train = np.load(os.path.join(args.input_path, 'X_train_logistic.npy'))
    y_train = np.load(os.path.join(args.input_path, 'y_train.npy'))

    clf = LogisticRegression(max_iter=10000).fit(x_train, y_train)
    
    y_score = clf.predict_proba(x_train)
    auroc = roc_auc_score(y_train, y_score[:, 1])
    auprc = average_precision_score(y_train, y_score[:, 1])

    pkl.dump(clf, open(os.path.join(args.output_path, 'logistic.pkl'), 'wb'))

    logger.info(f"[Test] AUROC {auroc}, AUPRC {auprc}")
    
    return