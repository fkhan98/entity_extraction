import os
import sys
import tqdm
import torch
import torch.nn as nn
from dataset import Dataset
from model import EntityExtractor
from transformers import AutoTokenizer

os.system('rm -rf ./logs.txt')

def train(model, pretrained_backbone, train_folder, val_folder, learning_rate, epochs, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_backbone)
    
    train, valid = Dataset(train_folder, tokenizer), Dataset(val_folder, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(valid, batch_size=batch_size)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        
    best_valid_loss = sys.maxsize
    for epoch_num in range(epochs):
        train_loss = 0
        valid_loss = 0
        model.train()
        print(f'Trainig loop for epoch: {epoch_num} has begun')
        for i, batch in enumerate(tqdm.tqdm(train_dataloader)):
            # print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['labels'].shape)
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            y_predict = model(input_ids=input_ids, attn_masks=attention_mask)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            labels = labels.view(-1)
            loss = criterion(y_predict, labels)
            optimizer.zero_grad()
            train_loss += loss.item()
            loss.backward()
            
        avg_train_loss_for_epoch = train_loss/(i+1)
        with open('./logs.txt', 'a') as f:
            f.write(f"Average training loss for epoch: {epoch_num} is: {avg_train_loss_for_epoch}\n")
            
        model.eval() 
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(val_dataloader)):
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                y_predict = model(input_ids=input_ids, attn_masks=attention_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                labels = labels.view(-1)
                loss = criterion(y_predict, labels)
                valid_loss += loss.item()
                
        avg_val_loss_for_epoch = valid_loss/(i+1)
        with open('./logs.txt', 'a') as f:
            f.write(f"Average validation loss for epoch: {epoch_num} is: {avg_val_loss_for_epoch}\n")
            
        path = './saved_model'
        save_path = os.path.join(path,f'{pretrained_backbone}_entity_extraction_model.pt')
        
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            
            
        if avg_val_loss_for_epoch < best_valid_loss:
            print(f'validation loss decreased from {best_valid_loss} to {avg_val_loss_for_epoch}, model being saved for epoch: {epoch_num}')
            with open('./logs.txt', 'a') as f:
                f.write(f'Validation loss decreased from {best_valid_loss} to {avg_val_loss_for_epoch}, model being saved for epoch: {epoch_num}\n')
            best_valid_loss = avg_val_loss_for_epoch
            torch.save(model.state_dict(), save_path)
        with open('./logs.txt', 'a') as f:
            f.write('======================================================================================\n')

        
if __name__ == "__main__":
    EPOCHS = 20
    # pretrained_backbone = "dmis-lab/biobert-v1.1"
    pretrained_backbone = "distilbert-base-uncased"
    model = EntityExtractor(pretrained_backbone=pretrained_backbone,freeze_bert=False)
    learning_rate = 1e-5
    batch_size = 8
    
    train(model=model, pretrained_backbone=pretrained_backbone, train_folder="./dataset/train", val_folder="./dataset/valid", learning_rate=learning_rate, epochs=EPOCHS, batch_size=batch_size)