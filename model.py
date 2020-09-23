import pandas as pd
import torch

from pprint import pprint

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from transformers import BertTokenizer, AdamW
from pytorch_lightning import LightningModule

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

from modeling_purifier import BertForSequenceClassification
from modeling_bert_badword import BertForBadWordFiltering


class Model(LightningModule):
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.bert = BertForSequenceClassification.from_pretrained(self.args.pretrained_model)
        # self.bert = BertForBadWordFiltering.from_pretrained(self.args.pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_tokenizer
            if self.args.pretrained_tokenizer
            else self.args.pretrained_model
        )

    def forward(self, **kwargs):
        return self.bert(**kwargs)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        loss, logits = self(input_ids=data, labels=labels)
        preds = logits.argmax(dim=-1)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'train_loss': loss.cpu().detach().numpy().tolist(),
            'train_acc': metrics[0],
            'train_precision': metrics[1],
            'train_recall': metrics[2],
            'train_f1': metrics[3],
        }
        if (batch_idx % self.args.report_cycle) == 0:
            print()
            pprint(tensorboard_logs)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        loss, logits = self(input_ids=data, labels=labels)
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def validation_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(outputs)

        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'val_loss': loss,
            'val_acc': metrics[0],
            'val_precision': metrics[1],
            'val_recall': metrics[2],
            'val_f1': metrics[3],
        }

        print()
        pprint(tensorboard_logs)
        return {'loss': _loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['document'] = df['document'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            padding='max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        return df

    def preprocess_text(self, text):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        text = pattern.sub(' ', str(text))
        text = url_pattern.sub('', text)
        text = text.strip()
        text = repeat_normalize(text, num_repeats=2)
        return self.tokenizer(text, max_length=self.args.max_length, truncation=True, return_tensors="pt")

    def train_dataloader(self):
        df = self.read_data(self.args.train_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )

    def val_dataloader(self):
        df = self.read_data(self.args.val_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )

    def save_model(self):
        self.bert.save_pretrained('bert-badword/')
        self.tokenizer.save_pretrained('tokenizer-badword/')
