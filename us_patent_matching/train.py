import shutil

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)

from us_patent_matching.dataset import USPatentDataset
from us_patent_matching.model import CustomModel
from us_patent_matching.utils import compute_metrics

CONFIG_NAME = 'baseline'

@hydra.main(config_path="config", config_name=CONFIG_NAME)
def main(config: DictConfig):

    train_df = pd.read_csv(f"{config.data_path}/train_folds.csv")
    titles = pd.read_csv(f'{config.data_path}/titles.csv')
    
    cpc_texts = torch.load(f"{config.data_path}/cpc_texts.pth")
    train_df['context_text'] = train_df['context'].map(cpc_texts)

    train_df['text'] = train_df['anchor'] + '[SEP]' + train_df['target'] + '[SEP]'  + train_df['context_text']

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    oof_df = pd.DataFrame()
    for fold in range(config.num_folds):
        
        tr_data = train_df[train_df['fold']!=fold].reset_index(drop=True)
        va_data = train_df[train_df['fold']==fold].reset_index(drop=True)
        tr_dataset = USPatentDataset(tr_data, tokenizer)
        va_dataset = USPatentDataset(va_data, tokenizer)
        
        # model = AutoModelForSequenceClassification.from_pretrained(config.model_path, num_labels=1)
        model = CustomModel(config)
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params=optimizer_parameters, lr=config.learning_rate)
        num_train_steps = int(len(train_df) / config.batch_size * config.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        args = TrainingArguments(
            output_dir=f"/tmp/uspppm",
            run_name=config.model_path,
            report_to='wandb',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            weight_decay=config.weight_decay,
            metric_for_best_model="pearson",
            load_best_model_at_end=True,
            fp16=True
        )
        

        trainer = Trainer(
            model,
            args,
            train_dataset=tr_dataset,
            eval_dataset=va_dataset,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        shutil.rmtree(f"/tmp/uspppm")
        trainer.save_model(f"{config.checkpoints}/{config.model_path}_{fold}")
        
        outputs = trainer.predict(va_dataset)
        predictions = outputs.predictions.reshape(-1)
        va_data['preds'] = predictions
        oof_df = pd.concat([oof_df, va_data])

    predictions = oof_df['preds'].values
    label = oof_df['score'].values
    eval_pred = predictions, label
    compute_metrics(eval_pred)
    oof_df.to_csv(f"{config.model_path}.csv", index=None)

if __name__ == "__main__":
    main()
