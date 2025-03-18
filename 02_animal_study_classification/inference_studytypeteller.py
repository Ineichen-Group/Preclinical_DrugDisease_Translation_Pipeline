import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import argparse

def load_model(model_path):
    # Determine the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model with correct device mapping
    model = torch.load(model_path, map_location=device)  # Ensures CPU compatibility
    model.to(device)  # Move model to correct device
    return model

def inference_on_new_data(new_data, model_path, model_name, batch_size=8, classification_type='multi'):
    """
    Perform inference on new, unlabeled data using a trained model and map label IDs to names.

    Parameters:
    - new_data (pd.DataFrame): DataFrame containing new data with columns 'PMID', 'journal_name', 'title', 'abstract', and optionally 'keywords'.
    - model_path (str): Path to the saved model.
    - model_name (str): Name of the model to load tokenizer and perform inference.
    - batch_size (int): Batch size for inference.

    Returns:
    - pd.DataFrame: DataFrame containing PMID, label predictions (as IDs and names), and confidence scores for each sample.
    """
    if classification_type == 'multi':
        # Label mapping dictionary
        label_mapping = {
            'Remaining': 0,
            'Non-systematic-review': 1,
            'Human-non-RCT-non-drug-intervention': 2,
            'Human-non-RCT-drug-intervention': 3,
            'Human-case-report': 4,
            'Animal-other': 5,
            'Animal-drug-intervention': 6,
            'Human-systematic-review': 7,
            'In-vitro-study': 8,
            'Human-RCT-non-drug-intervention': 9,
            'Animal-non-drug-intervention': 10,
            'Human-RCT-drug-intervention': 11,
            'Clinical-study-protocol': 12,
            'Human-RCT-non-intervention': 13
        }
    else:
        label_mapping = {
            'Remaining': 0,
            'Animal': 1
        }
    # Reverse the mapping to get label names from IDs
    id_to_label = {v: k for k, v in label_mapping.items()}

    # Load model and tokenizer
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Preprocess and tokenize the new data
    def concatenate_text(row):
        text_parts = [str(row['journal_name']), str(row['title']), str(row['abstract'])]
        keywords = row.get('keywords', None)
        if pd.notna(keywords):
            keywords_list = keywords.split('|')
            text_parts.extend(keywords_list)
        return ' '.join(text_parts)

    new_data['text'] = new_data.apply(concatenate_text, axis=1)
    encodings = tokenizer(new_data['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    confidences = []

    # Perform inference
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            confidences.extend(probs.max(dim=1).values.cpu().numpy())

    # Map predicted label IDs to label names
    label_ids = predictions
    label_names = [id_to_label[label_id] for label_id in label_ids]

    # Construct DataFrame for output with PMID, label ID, label name, and confidence
    result_df = pd.DataFrame({
        'PMID': new_data['PMID'],
        'label_id': label_ids,       # Add label IDs
        'label_name': label_names,   # Add label names
        'confidence': confidences
    })

    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model.")
  
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="The name or path of the HuggingFace model to use. For example, 'michiyasunaga/BioLinkBERT-base'."
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default="./scripts/inference/best_model_PubMedBERT_multi.pt",
        help="Path to the fine-tuned .pt file of the HuggingFace model."
    )
    parser.add_argument(
        "--pubmed_file",
        type=str,
        default="./scripts/inference/pmid_contents_chunk_0.txt",
        help="File with PubMed content."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./scripts/inference/predictions_chunk_0.txt",
        help="File name to save predictions, e.g. ./model_predictions/neuro_pubmed/predictions_chunk_0.txt."
    )

    args = parser.parse_args()
   
    model_name = args.model_name_or_path
    model_path = args.trained_model_path
    model_name_clean = model_name.split("/")[1]
    
    input_file_path = args.pubmed_file
    out_file = args.output_file

    headers = ["PMID", "year", "journal_name", "title", "abstract", "publication_type"]
    
    separator = "|||"

    # Read and process the file manually to handle the multi-character delimiter
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line using the delimiter and add to the list
            fields = line.strip().split(separator)
            if len(fields) == len(headers):  # Ensure the correct number of fields
                data.append(fields)
            else:
                print("Nr. of fields not equal nr. of expected header columns!")

    # Convert the processed data into a DataFrame
    new_data = pd.DataFrame(data, columns=headers)

    #new_data = pd.read_csv(input_file_path, sep=r'|||', names=headers,  engine='python')  # Change 'sep' if files use a different delimiter

    # Perform inference
    results = inference_on_new_data(new_data, model_path, model_name, classification_type='binary')
    results.to_csv(out_file, index=False)