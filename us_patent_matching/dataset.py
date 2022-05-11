import torch
from torch.utils.data import Dataset


class USPatentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.texts = df['text'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = df['score'].values
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.tokenizer(
            self.texts[item],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=False,
            #return_tensors="pt",
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return {
            **inputs,
            'labels': torch.tensor(self.labels[item], dtype=torch.float)
        }
