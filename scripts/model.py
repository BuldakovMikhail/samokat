from torch import nn
from transformers import BertTokenizer, BertModel


class BertWithClassifierMixUp(nn.Module):
    def __init__(self, linear_size):
        super(BertWithClassifierMixUp, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                in_features=self.bert.pooler.dense.out_features,
                out_features=linear_size,
            ),
            nn.BatchNorm1d(num_features=linear_size),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=linear_size, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, tokens1, attention_mask1, tokens2, attention_mask2, lam):
        bert_output1 = self.bert(input_ids=tokens1, attention_mask=attention_mask1)
        bert_output2 = self.bert(input_ids=tokens2, attention_mask=attention_mask2)

        bert_output = lam * bert_output1[1] + (1.0 - lam) * bert_output2[1]

        y = self.head(bert_output)
        return y

    def freeze_bert(self):
        for param in self.bert.named_parameters():
            param[1].requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.named_parameters():
            param[1].requires_grad = True
