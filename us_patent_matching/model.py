import torch
import torch.nn as nn
from torch.nn import MSELoss
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class TransformerHead(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, nhead=8, num_targets=1):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead), num_layers=num_layers
        )
        self.row_fc = nn.Linear(in_features, num_targets)
        self.out_features = out_features

    def forward(self, x):
        out = self.transformer(x)
        out = self.row_fc(out)
        return out

class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model_path, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.model_path)
 
        self.feature_extractor = AutoModelForTokenClassification.from_pretrained(cfg.model_path)
        in_features = self.feature_extractor.classifier.in_features
        self.attention = TransformerHead(in_features=in_features, out_features=cfg.max_len, num_layers=1, nhead=8, num_targets=1)
        self.fc_dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1024, 1)
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
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        
        return feature

    def forward(self, labels=None, **inputs):
        feature = self.feature(**{i:inputs[i] for i in inputs if i!='labels'})
        logits = self.fc(self.fc_dropout(feature))

        loss=None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
