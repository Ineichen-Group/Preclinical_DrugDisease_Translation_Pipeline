import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pandas as pd
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt


def setup_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'), mode='a') # do not overwrite log
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_short_model_name(model_name):
    model_names = {
        'bert-base-uncased': 'bert-base',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': 'PubMedBERT',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract': 'BiomedBERT',
        'allenai/scibert_scivocab_uncased': 'Scibert',
        'dmis-lab/biobert-v1.1': 'biobert',
        'michiyasunaga/BioLinkBERT-base': 'BioLinkBERT',
        'emilyalsentzer/Bio_ClinicalBERT': 'Bio_ClinicalBERT'}
    if model_name in model_names.keys():
        return model_names[model_name]
    else:
        raise ValueError("Invalid model name. Cannot be mapped to short name.")


def load_data_splits(data_dir, col_name, tokenizer_name, batch_size):
    # set paths to data splits and create dfs
    train_file = os.path.join(data_dir, 'train.csv')
    val_file = os.path.join(data_dir, 'val.csv')
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # TODO add keywords to concatenation
    def concatenate_text(row):
        text_parts = [str(row['journal_name']), str(row['title']), str(row['abstract'])]
        keywords = row.get('keywords')
        if keywords and pd.notna(keywords):
            keywords_list = keywords.split('|')
            text_parts.extend(keywords_list)
        return ' '.join(text_parts)

    train_df['text'] = train_df.apply(concatenate_text, axis=1)
    val_df['text'] = val_df.apply(concatenate_text, axis=1)
    print(train_df.text.head())
    print(val_df.text.head())
 
    # tokenize and encode textual data
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_encodings = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')
    val_encodings = tokenizer(val_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')
 
    # encode labels
    train_labels = torch.tensor(train_df[col_name].values)
    val_labels = torch.tensor(val_df[col_name].values)

    # create encoded dataset
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    return DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size), DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)


def train_model(model_name, tokenizer_name, col_name, num_labels, epochs, patience, batch_size, learning_rate, weight_decay, data_dir, save_dir, log_dir, classification_type, logger):

    model_dir = os.path.join(save_dir, get_short_model_name(model_name))
    os.makedirs(model_dir, exist_ok=True)

    train_dataloader, val_dataloader = load_data_splits(data_dir, col_name, tokenizer_name, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(total_steps * 0.1) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    best_val_loss = float('inf')
    best_model_path = None
    no_improvement_count = 0  # Initialize the counter for epochs without improvement
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        tqdm_train_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in tqdm_train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, batch[2])  # Compute cross-entropy loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            tqdm_train_dataloader.set_postfix({'training_loss': train_loss / (tqdm_train_dataloader.n + 1)})  # Update the progress bar with training loss

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, batch[2])  # Compute cross-entropy loss
                val_loss += loss.item()

                # Append predictions and labels for F1-score computation
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch[2].cpu().numpy())

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        # save checkpoints into model dir
        checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_dir, f"best_model_{get_short_model_name(model_name)}_{classification_type}.pt")
            torch.save(model, best_model_path)
            no_improvement_count = 0
            logger.info(f"Saving to {best_model_path}")
        else:
            no_improvement_count += 1


        f1 = f1_score(val_labels, val_preds, average='weighted')
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Validation Loss: {val_loss}")
        logger.info(f"Validation F1-score: {f1}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Validation Loss: {val_loss}")
        print(f"Validation F1-score: {f1}")
        
        # Early stopping / patience
        if no_improvement_count >= patience:
            logger.info(f"Early stopping as validation loss didn't improve for {patience} consecutive epochs.")
            print((f"Early stopping as validation loss didn't improve for {patience} consecutive epochs."))
            break

    
    ################# Plot train and val losses #################
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss: {get_short_model_name(model_name)}')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f'train_loss.png'))
    plt.show()
    plt.close()

    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss: {get_short_model_name(model_name)}')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f'val_loss.png'))
    plt.show()
    plt.close()


def main(classification_type):
    # define input and output dirs
    data_dir = "./data/study_type_teller/9-1-0_all_classes_enriched"
    save_dir = f"./models/transformers/models_binary"
    log_dir = f"./models/transformers/logs"
 
    # TODO modify the list of models, if needed
    models_to_fine_tune = [
                            #'bert-base-uncased',
                            #'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                            #'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                            'allenai/scibert_scivocab_uncased',
                            #'dmis-lab/biobert-v1.1',
                            #'michiyasunaga/BioLinkBERT-base',
                            #'emilyalsentzer/Bio_ClinicalBERT',
                            ]

    if classification_type == 'binary':
        col_name = 'binary_label'
        num_labels = 2
    elif classification_type == 'multi':
        col_name = 'multi_label'
        num_labels = 14
    else:
        raise ValueError("Invalid classification_type. Must be either 'binary' or 'multi'.")
    
    # set up logger
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # iterate over models and finetune them
    for model_name in models_to_fine_tune:
        logger.info(f"\n\n************** Fine-tuning {model_name}: {classification_type} **************")
        print((f"\n\n************** Fine-tuning {model_name}: {classification_type} **************"))
        train_model(
            model_name=model_name,
            tokenizer_name=model_name, # TODO double-check they correspond
            col_name=col_name,
            num_labels = num_labels,
            epochs=10,
            patience=4,
            batch_size=8,
            learning_rate=5e-5,
            weight_decay=5e-4,
            data_dir=data_dir,
            save_dir=save_dir,
            log_dir=log_dir,
            classification_type=classification_type,
            logger=logger
        )


if __name__ == "__main__":
    # TODO select classification type (multi, binary)
    main(classification_type='binary')  
