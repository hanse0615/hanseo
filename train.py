import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from setproctitle import setproctitle
setproctitle('Jungseob-Test')

import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import evaluate
from preprocess import get_preprocessed_binary

import wandb


rouge_ = evaluate.load("rouge")
bleu_ = evaluate.load("bleu")
meteor_ = evaluate.load("meteor")


class BrailleDataset(Dataset):
    def __init__(self, text_list, braille_list, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        self.references = text_list
        
        for text, braille in tqdm(zip(text_list, braille_list), total=len(text_list)):
            text_encoding = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            braille_encoding = self.tokenizer(braille, return_tensors="pt", max_length=1024, truncation=True)
            self.data.append((text_encoding, braille_encoding))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
def collate_fn(batch):
    input_ids = []
    attention_masks = []
    labels = []

    for text_encoding, braille_encoding in batch:
        input_ids.append(braille_encoding['input_ids'].squeeze(0))
        attention_masks.append(braille_encoding['attention_mask'].squeeze(0))
        labels.append(text_encoding['input_ids'].squeeze(0))

    # Padding input_ids and attention_masks
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_masks
    }

    return inputs, labels


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def calculate_metrics(pred, korean):    
    bleu_scores = bleu_.compute(
        predictions=pred,
        references=korean
    )['bleu']
    
    rougeL = rouge_.compute(
        predictions=pred,
        references=korean
    )["rougeL"]

    meteor_scores = meteor_.compute(
        predictions=pred,
        references=korean
    )['meteor']
    
    return bleu_scores, rougeL, meteor_scores

initialize_encoder = True
initialize_encoder_layer = False

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")

if initialize_encoder:
    # 이진수 조합 토큰 추가
    binary_tokens = [
        '000000', '000001', '000010', '000011', '000100', '000101', '000110', 
        '000111', '001000', '001001', '001010', '001011', '001100', '001101', 
        '001110', '001111', '010000', '010001', '010010', '010011', '010100', 
        '010101', '010110', '010111', '011000', '011001', '011010', '011011', 
        '011100', '011101', '011110', '011111', '100000', '100001', '100010', 
        '100011', '100100', '100101', '100110', '100111', '101000', '101001', 
        '101010', '101011', '101100', '101101', '101110', '101111', '110000', 
        '110001', '110010', '110011', '110100', '110101', '110110', '110111', 
        '111000', '111001', '111010', '111011', '111100', '111101', '111110', 
        '111111'
    ]
    # 이진수 조합 토큰 추가
    tokenizer.add_tokens(binary_tokens)
    model.resize_token_embeddings(len(tokenizer))


braille_data = get_preprocessed_binary()[:-1]
korean_text = open("./data/preprocess_backbackback.txt", "r", encoding="utf-8").readlines()
korean_text = [line.replace(" 뷁뷁뷁뷁뷁\n", "") for line in korean_text]

# [:1000]
full_dataset = BrailleDataset(
    text_list=korean_text,
    braille_list=braille_data,
    tokenizer=tokenizer
)
print(braille_data[4])
print()
print(braille_data[5])
print()
print(braille_data[6])
print()

train_size = int(0.99 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
references = val_dataset.dataset.references

# encoder layer 두 개만 선택
if initialize_encoder_layer:
    # model.config.encoder_layers = 2
    model.model.encoder.layers = model.model.encoder.layers[:2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = Adam(model.parameters(), lr=3e-5)
# lr_scheduler = StepLR(optimizer, step_size=1000, gamma=0.05)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
epochs = 13
validation_steps = 500
step = 0
total_train_steps = len(train_dataloader) * epochs

early_stopping_patience = 3
no_improvement = 0
best_bleu_score = -1

for inputs, label in train_dataloader:
    print(inputs['input_ids'][0])
    # print(inputs['input_ids'][1])
    # print(inputs['input_ids'][2])

wandb.init(
    project='braille',
    name='hyunwoongko_kobart',
)


for epoch in range(epochs):
    epoch_step = 0
    for (inputs, labels) in train_dataloader:
        # try:      
        outputs = model(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            labels=labels.to(device)
        )
            
        loss = outputs.loss
        wandb.log(
            data={
                f"train/epoch": epoch,
                f"train/loss": loss.item(),
                "lr": get_lr(optimizer),
            },
            step=step,
        )
        
        print(
            f"EPOCH: {epoch}/{epochs} ({epoch_step}/{len(train_dataloader)}), STEP: {step}/{total_train_steps}, LOSS:{loss.item():.4f}"
        )
        
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step() 
                
        
        if step % validation_steps == 0:
            model.eval()
            print('Generate...')
            val_loss_list = []
            generated_text_list = []
            reference_list = []
            random_idx = random.randint(0, len(val_dataloader) - 1)  # 랜덤 인덱스 생성
            
            with torch.no_grad():
                for idx, (inputs, labels) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    outputs = model(
                        input_ids=inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        labels=labels.to(device)
                    ).loss.item()
                    val_loss_list.append(outputs)
                    
                    # val_loss_list.append(
                    #     model(
                    #         input_ids=inputs['input_ids'].to(device),
                    #         attention_mask=inputs['attention_mask'].to(device),
                    #         labels=labels.to(device)
                    #     ).loss.item()
                    # )
                    
                    outputs = model.generate(
                        input_ids=inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        min_length=5,
                        max_length=45,
                        num_beams=2,
                    )
                    
                    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    generated_text_list.extend(output_texts)
                    reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    reference_list.extend(reference_texts)       
                                 
            print("Validation Results")
            print('---'*15)
            print(f"Generated text: {generated_text_list[random_idx]}")
            print(f"Reference: {reference_list[random_idx]}")
            print('---'*15)
            print(f"Generated text 2: {generated_text_list[0]}")
            print(f"Reference 2: {reference_list[0]}")
            print('---'*15)
            bleu, rouge_l, meteor = calculate_metrics(generated_text_list, reference_list)
            print(f"BLEU: {bleu}")
            print(f"ROUGE-L: {rouge_l}")
            print(f"METEOR: {meteor}")
            print('---'*15)
            
            wandb.log(
                data={
                    f"valid/BLEU": bleu,
                    f"valid/ROUGE-L": rouge_l,
                    f"valid/METEOR": meteor,
                    f"valid/loss": sum(val_loss_list)/len(val_loss_list),
                },
                step=step,
            )
            
            # if bleu >= best_bleu_score:
            #     best_bleu_score = bleu
            #     no_improvement = 0
            # else:
            #     no_improvement += 1
            #     if no_improvement >= early_stopping_patience:
            #         print("Early stopping due to no improvement in BLEU score")
            #         break
            

            # if no_improvement >= early_stopping_patience:
            #     print("Early stopping due to no improvement in BLEU score")
            #     break
            
        step += 1
        epoch_step += 1
    
# 모델 저장
model.save_pretrained("./ckpt/kobart_6_layer_and_13_epochs_re-tokenizer")
tokenizer.save_pretrained("./ckpt/kobart_6_layer_and_13_epochs_re-tokenizer")