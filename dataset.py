import glob
import json
import torch
from transformers import AutoTokenizer

TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    }
}

with open("./labels.json", "r") as json_file:
    label_map = json.load(json_file)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer):
        self.data_files = glob.glob(f"{file_path}/*.txt")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_files)

    
    def __getitem__(self, index, sequence_len = 512):
        file_path = self.data_files[index]
        with open(file_path, "r") as f:
            data = f.readlines()
            
        x = [TOKEN_IDX['bert']['START_SEQ']]
        label = [0]
        y_mask = [1]
        # print(data)
        for row in data:
            entity_tag = row.rstrip().split("\t")
            word = entity_tag[0]
            tag = entity_tag[1]
            
            tokens = self.tokenizer.tokenize(word)

            for token in tokens:
                x.append(self.tokenizer.convert_tokens_to_ids(token))
                label.append(label_map[tag])
                y_mask.append(1)
       
        x.append(TOKEN_IDX['bert']['END_SEQ'])
        label.append(0)
        y_mask.append(1)
        
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX['bert']['PAD'] for _ in range(sequence_len - len(x))]
            label = label + [0 for _ in range(sequence_len - len(label))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        
        # print({
        #     'input_ids': torch.tensor(x),
        #     'labels': torch.tensor(label),
        #     'attention_mask': torch.tensor(y_mask)
        # })
        
        return {
            'input_ids': torch.tensor(x),
            'labels': torch.tensor(label),
            'attention_mask': torch.tensor(y_mask)
        }
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    dummy_dataset = Dataset(file_path="./dataset/train", tokenizer=tokenizer)
    
    dummy_dataset.__getitem__(4)