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
        self.masking_id = self.tokenizer("*", max_length=self.args.max_length, truncation=True, return_tensors="pt")['input_ids'][0][1:-1].tolist()

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
        return text, self.tokenizer(text, max_length=self.args.max_length, truncation=True, return_tensors="pt"), self.tokenizer.tokenize(text)

    def inference(self, text):
        print(text)

        flag = False

        text, inputs, text_tokenized = self.preprocess_text(text)

        while True:
            with torch.no_grad():
                logits, cls_info = self(**inputs)
                pred = logits[0].argmax()

                if not flag:
                    logits_first, pred_first = logits[0], pred
                    flag = True

                if pred:
                    input_ids = inputs['input_ids'].numpy()[0].tolist()
                    toxic_ids = torch.argsort(cls_info['probs'][0][0][0][1:-1], descending=True)

                    no_word = False
                    for idx in range(len(toxic_ids)):
                        if input_ids[toxic_ids.tolist()[idx]+1] != self.masking_id[0]:
                            no_word = True
                            break

                    if not no_word:
                        break

                    toxic_id = toxic_ids.tolist()[idx]

                    masking_text = self.tokenizer.decode(input_ids[toxic_id+1], skip_special_tokens=True).replace("#", "").replace(" ", "")

                    input_ids = input_ids[:toxic_id+1] + self.masking_id*len(masking_text) + input_ids[toxic_id+2:]

                    text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    print(text)

                    inputs = {
                             'input_ids': torch.LongTensor(input_ids).unsqueeze(0),
                             'token_type_ids': torch.LongTensor([[0]*len(input_ids)]),
                             'attention_mask': torch.LongTensor([[1]*len(input_ids)])
                             }
                else:
                    print(text)
                    break
        return torch.nn.functional.softmax(logits_first).cpu().numpy(), pred_first.cpu().numpy(), text
