import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModel, AutoTokenizer, Trainer,
                          TrainingArguments)
from transformers.modeling_outputs import SequenceClassifierOutput


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class config:
    data_path = '/root/us_patent_matching/input' #'../input/us-patent-phrase-to-phrase-matching/'
    m_path= '/root/us_patent_matching/checkpoints/anferico/bert-for-patents' #"../input/pppm-deberta-v3-large-closing-the-cv-lb-gap/"
    batch_size=32
    num_workers=4
    num_folds=5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(self.cfg.model_path, output_hidden_states=True)
        self.model = AutoModel.from_config(self.config)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.fc_dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, **inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        # import IPython; IPython.embed(); exit(1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, labels=None, **inputs):
        feature = self.feature(**{i:inputs[i] for i in inputs if i!='labels'})
        logits = self.fc(self.fc_dropout(feature))
        loss=None
        if labels:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class USPatentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.texts = df['text'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.tokenizer(
            self.texts[item],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=False,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return {
            **inputs
        }

test_df = pd.read_csv(f"{config.data_path}/test.csv")
titles = pd.read_csv(f'{config.data_path}/titles.csv')

cpc_texts = torch.load(f"{config.data_path}/cpc_texts.pth")
test_df['context_text'] = test_df['context'].map(cpc_texts)

test_df['text'] = test_df['anchor'] + '[SEP]' + test_df['target'] + '[SEP]'  + test_df['context_text']

tokenizer = AutoTokenizer.from_pretrained(f'{config.m_path}_0')
test_dataset = USPatentDataset(test_df, tokenizer)

predictions = []
for fold in range(config.num_folds):
    config.model_path = f"{config.m_path}_{fold}"
    model = CustomModel(config)
    model.load_state_dict(torch.load(f"{config.model_path}/pytorch_model.bin", map_location=config.device))

    args = TrainingArguments(
        output_dir=".",
        fp16=True
    )

    trainer = Trainer(
        model,
        args,
        tokenizer=tokenizer
    )
    prediction = sigmoid(trainer.predict(test_dataset).predictions.reshape(-1))

    predictions.append(prediction)

predictions = np.mean(predictions, axis=0)

test_df['score'] = predictions
test_df[['id', 'score']].to_csv('submission.csv', index=False)
