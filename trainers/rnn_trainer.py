import os
import logging
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from dataset import EHRDataset
from models.rnn import EHRRNN

logger = logging.getLogger(__name__)


class RNNTrainer:
    def __init__(self, args):
        self.args = args
        self.datasets, self.dataloader = dict(), dict()
        self.datasets["train"] = EHRDataset(self.args, 'train')
        self.datasets["test"] = EHRDataset(self.args, 'test')
        self.dataloader["train"] = DataLoader(self.datasets["train"], collate_fn=self.datasets["train"].collator, num_workers=8, batch_size=self.args.batch_size, shuffle=True)
        self.dataloader["test"] = DataLoader(self.datasets["test"], collate_fn=self.datasets["test"].collator, num_workers=8, batch_size=self.args.batch_size, shuffle=True)

        self.device = torch.device(f"cuda:{self.args.device_num}" if torch.cuda.is_available() else "cpu")

        self.model = EHRRNN(self.args)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model.to(self.device)
        self.best_auroc = 0

        wandb.init(
            project=self.args.wandb_project_name,
            entity=self.args.wandb_entity_name,
            config=self.args,
            reinit=True
        )
        wandb.run.name = f"rnn_{self.args.hidden_dim}_{self.args.lr}"
    

    def train(self):

        for i in range(self.args.epoch):
            
            logger.info(f"Start Training Epoch {i}...")

            self.model.train()
            y_true = []
            y_score = []
            total_loss = 0

            for step, sample in enumerate(tqdm.tqdm(self.dataloader["train"])):
                
                self.optimizer.zero_grad()

                for key in sample["input"]:
                    if key != "s_len":
                        sample["input"][key] = sample["input"][key].to(self.device)
                
                output = self.model(**sample["input"])
                loss = self.criterion(output, torch.tensor(sample["labels"]).to(self.device))
                
                probs = nn.functional.softmax(output, dim=-1).cpu().detach()
                y_true.extend(sample["labels"])
                y_score.extend(probs[:, 1].tolist())
                total_loss += loss

                loss.backward()
                self.optimizer.step()

            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)

            wandb.log({'epoch': i, 'train/auroc': auroc, 'train/auprc': auprc, 'train/loss': total_loss / (step + 1)})
            logger.info(f"[Train Epoch {i}] AUROC {auroc}, AUPRC {auprc}, Loss {total_loss / (step + 1)}")

            self.eval()
        
        wandb.finish(0)


    def eval(self):
        
        self.model.eval()
    
        with torch.no_grad():
            
            logger.info("Testing...")

            y_true = []
            y_score = []
            total_loss = 0

            for step, sample in enumerate(tqdm.tqdm(self.dataloader["test"])):

                for key in sample["input"]:
                    if key != "s_len":
                        sample["input"][key] = sample["input"][key].to(self.device)

                output = self.model(**sample["input"])
                loss = self.criterion(output, torch.tensor(sample["labels"]).to(self.device))

                probs = nn.functional.softmax(output, dim=-1).cpu().detach()
                y_true.extend(sample["labels"])
                y_score.extend(probs[:, 1].tolist())
                total_loss += loss

            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)

            wandb.log({'test/auroc': auroc, 'test/auprc': auprc, 'test/loss': total_loss / (step + 1)})
            logger.info(f"[Test] AUROC {auroc}, AUPRC {auprc}, Loss {total_loss / (step + 1)}")

            if auroc > self.best_auroc:
                save_path = os.path.join(self.args.output_path, "rnn.pt")
                logger.info(f"Saved model to {save_path}...")
                torch.save(self.model.state_dict(), save_path)

