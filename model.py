import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class EntityExtractor(nn.Module):
    def __init__(self,pretrained_backbone,freeze_bert=False):
        super(EntityExtractor, self).__init__()
        with open("./labels.json", "r") as json_file:
            self.output_lables = json.load(json_file)
        self.output_dim = len(self.output_lables)
        self.lm_layer = AutoModel.from_pretrained(pretrained_backbone)
        
        if freeze_bert:
            for p in self.lm_layer.parameters():
                p.requires_grad = False
                
        self.linear = nn.Linear(in_features=768, out_features=self.output_dim)
    
    def forward(self, input_ids, attn_masks):
        x = self.lm_layer(input_ids, attention_mask=attn_masks)[0]
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = EntityExtractor()
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    input_sentence = "SOCIAL HISTORY:  Lives with parents and with two siblings, one 18-year-old and the other is 14-year-old in house, in Corrales. They have animals, but outside the house and father smokes outside house. No sick contacts as the mother said."
    input_ids = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
    # print(input_ids)
    output = model(input_ids['input_ids'],input_ids['attention_mask'])
    print(output.shape)


