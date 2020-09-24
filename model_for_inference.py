import torch

from transformers import BertForSequenceClassification, BertTokenizer
from pytorch_lightning import LightningModule

import re
import emoji
from soynlp.normalizer import repeat_normalize


class Model(LightningModule):
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.bert = BertForSequenceClassification.from_pretrained(self.args.pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_tokenizer
            if self.args.pretrained_tokenizer
            else self.args.pretrained_model
        )

    def forward(self, **kwargs):
        return self.bert(**kwargs)

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

    def inference(self, text):
        text = self.preprocess_text(text)
        with torch.no_grad():
            logits = self(**text)[0]
            pred = logits[0].argmax()
        return logits[0].cpu().numpy(), pred.cpu().numpy()
