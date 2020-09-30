import torch

from transformers import BertTokenizer
from pytorch_lightning import LightningModule

import re
import emoji
from soynlp.normalizer import repeat_normalize

from modeling_purifier import BertForSequenceClassification


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
        print(text)
        return self.tokenizer(text, max_length=self.args.max_length, truncation=True, return_tensors="pt"), self.tokenizer.tokenize(text)

    def inference(self, text):
        inputs, text = self.preprocess_text(text)
        print(inputs)
        print(text)

        dec = self.tokenizer.decode(inputs['input_ids'].numpy()[0], skip_special_tokens=True)
        print(dec)
        # cls_info의 'prob'에서  [0][0]가 이제 문장에서 확률, 긍까 그걸 리턴?
        with torch.no_grad():
            logits, cls_info = self(**inputs)
            print(logits)
            print(cls_info.keys())
            pred = logits[0].argmax()
        return logits[0].cpu().numpy(), pred.cpu().numpy()
