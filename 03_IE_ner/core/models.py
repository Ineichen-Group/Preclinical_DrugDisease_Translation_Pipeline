from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer
import pandas as pd
import re
import torch
from datasets import load_dataset
from .bert_helper import get_label_list, tokenize_and_align_labels, tokenize_and_align_labels_with_overflow
import numpy as np
from ast import literal_eval
from pathlib import Path


class NERModel:
    def __init__(self, model_type, model_name, model_path=None, entity_class_names_dict=None, use_custom_entities_grouping = False):
        self.model_name = model_name
        if "/" in model_name:
            self.model_name_short = self.model_name.split("/")[1]
        else:
            self.model_name_short = self.model_name
        self.model_path = model_path
        self.model_type = model_type
        self.return_words_only = False
        self.entity_class_names_dict = entity_class_names_dict
        self.normalize_pred_representation = True  # used for the hugging face models - it removes the additional information like prediction confidence that other models don't provide
        self.use_custom_entities_grouping = use_custom_entities_grouping

        self.load_model()


    def load_model(self):
        if self.model_type == "spacy":
            self.nlp = spacy.load(self.model_path)
        elif self.model_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.config = AutoConfig.from_pretrained(self.model_path)
            
            if self.use_custom_entities_grouping:
                aggregation_strategy = "none"
            else:
                aggregation_strategy = "simple"
            
            self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                                aggregation_strategy=aggregation_strategy)  # grouped entities False to analyze the tokenization of the models
        elif self.model_type == "regex":
            pass
        else:
            raise ValueError("Wrong model type. Allowed values are spacy, regex, and huggingface.")

    def bert_predict_bio_format(
        self,
        ds_path_train,
        ds_path_test,
        text_column_name="tokens",
        label_column_name="ner_tags",
        label_all_tokens=False,
        padding=True,
        max_length=512,
        stride=128
    ):
        """
        Run predictions in BIO format and return a DataFrame
        with doc_id, tokens, gold labels, and predicted labels aligned 1:1.
        Handles long sequences by splitting into chunks and merging back.
        """
        model = self.model
        
        def ensure_list_format(example, label_column_name="ner_tags"):
            if isinstance(example[label_column_name], str):
                try:
                    example[label_column_name] = literal_eval(example[label_column_name])
                except Exception:
                    example[label_column_name] = example[label_column_name].split()
            return example

        # Load datasets
        data_files = {"train": ds_path_train, "test": ds_path_test}
        raw_datasets = load_dataset("json", data_files=data_files)
        raw_datasets = raw_datasets.map(
            ensure_list_format,
            fn_kwargs={"label_column_name": label_column_name},
            batched=False
        )

        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
        id_to_label = {i: l for i, l in enumerate(label_list)}        

        predict_dataset = raw_datasets["test"].map(
            tokenize_and_align_labels_with_overflow,
            batched=False,
            desc="Tokenizing prediction dataset with stride",
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "text_column_name": text_column_name,
                "label_column_name": label_column_name,
                "label_to_id": label_to_id,
                "label_all_tokens": label_all_tokens,
                "padding": padding,
                "max_length": max_length,
                "stride": stride
            }
        )

        # Run prediction
        trainer = Trainer(model=model, tokenizer=self.tokenizer)
        results = trainer.predict(predict_dataset)
        predictions = np.argmax(results.predictions, axis=2)

        # Merge predictions back to full documents
        sample_mapping = predict_dataset["overflow_to_sample_mapping"]

        merged_preds = {i: [] for i in range(len(raw_datasets["test"]))}
        for i, pred_seq in enumerate(predictions):
            doc_idx = sample_mapping[i]
            word_ids = predict_dataset[i]["word_ids"]
            prev_word_id = None
            for p, w_id in zip(pred_seq, word_ids):
                if w_id is None or w_id == prev_word_id:
                    continue
                merged_preds[doc_idx].append(id_to_label[p])
                prev_word_id = w_id

        # Build DataFrame with original fields
        df = pd.DataFrame({
            "doc_id": raw_datasets["test"]["doc_id"],
            "tokens": raw_datasets["test"][text_column_name],
            "gold_labels": raw_datasets["test"][label_column_name],
            "pred_labels": [merged_preds[i] for i in range(len(raw_datasets["test"]))],
        })

        # Sanity check: lengths should match
        for i, row in df.iterrows():
            if len(row["tokens"]) != len(row["pred_labels"]):
                print(f"⚠️ Length mismatch in doc {row['doc_id']}: "
                    f"tokens={len(row['tokens'])}, preds={len(row['pred_labels'])}")

        return df


    def annotate(self, file_path, source_column, sep=","):
        # If caller passed a DataFrame in the "file_path" slot
        if isinstance(file_path, pd.DataFrame):
            df = file_path.copy()

        # If caller passed a path-like (str / Path)
        elif isinstance(file_path, (str, Path)):
            file_path = str(file_path)

            if file_path.endswith(".jsonl"):
                df = pd.read_json(file_path, lines=True)

            elif file_path.endswith(".json"):
                df = pd.read_json(file_path)

            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path, sep=sep)

            else:
                raise ValueError(f"Unsupported file type: {file_path}")

        else:
            raise TypeError(
                f"file_path must be a pandas.DataFrame or a path (str/Path), got {type(file_path)}"
            )

        if self.normalize_pred_representation:
            predictions_col_name = "ner_prediction_{}_normalized".format(self.model_name_short)
        else:
            predictions_col_name = "ner_prediction_{}".format(self.model_name_short)

        df[predictions_col_name] = df.apply(lambda row: self.infer_ner(row[source_column]), axis=1)
        return df

    def annotate_parallel(self, file_path, source_column, sep=","):
        df = pd.read_csv(file_path, sep=sep)
        num_processes = 10
        chunks = [df[i:i + len(df) // num_processes] for i in
                  range(0, len(df), len(df) // num_processes)]

        with Pool(num_processes) as pool:
            # Apply the function to each chunk in parallel
            results = pool.starmap(self.infer_ner_chunk,
                                   [(chunk, source_column) for chunk in chunks])
        return pd.concat(results)

    def infer_ner_chunk(self, chunk, text_source_column):
        predictions_col_name = "ner_prediction_{}".format(self.model_name_short)
        chunk[predictions_col_name] = chunk[text_source_column].apply(self.infer_ner)
        if self.model_type == "huggingface":
            self.normalize_pred_representation = True
            predictions_col_name = "ner_prediction_{}_normalized".format(self.model_name_short)
            chunk[predictions_col_name] = chunk[text_source_column].apply(self.infer_ner)
            # predictions_col_name = "ner_prediction_{}_bio".format(self.model_type)
            # chunk[predictions_col_name] = chunk[text_source_column].apply(self.ner_bert_bio_output)
        return chunk

    def ner_bert_bio_output(self, sentence):
        # sentence = normalizer.normalize(sentence)

        model, tokenizer, labels = self.model, self.tokenizer, list(self.config.label2id.keys())
        #### check length limit ####
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        num_tokens = len(tokenized_sentence)
        if num_tokens > 512:
            tokenized_sentence = tokenized_sentence[:510]
            sentence = self.tokenizer.convert_tokens_to_string(tokenized_sentence)
        #### check length limit ####

        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model(inputs)[0]
        predictions = torch.argmax(outputs, axis=2)
        predictions = [(token, labels[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())]

        filtered_predictions = []
        prev_token, prev_label = None, None

        # dealing with the misaligned tokenizations
        # TODO: seems too manual... trying to fit to the the prodigy tokenization here
        for token, label in predictions:
            if prev_token == 'cann' and token == '##ot' or (
                    token == "##mg"):  # spacy splits connot into "can" and "not", two tokens
                filtered_predictions.append(label)
            elif token.startswith("##®") or (not token.startswith('##') and token not in ['[SEP]',
                                                                                          '[CLS]']):  # ignore the word pieces that would inflate the array
                filtered_predictions.append(label)
            prev_token, prev_label = token, label

        return filtered_predictions

    def infer_ner(self, sent, tokenized_sent=None):
        if self.model_type == "spacy":
            doc = self.nlp(sent)
            entities = []
            for ent in doc.ents:
                if self.return_words_only:  # TODO where is the right place?
                    entities.append(ent.text)
                else:
                    ent_label = ent.label_
                    if ent_label == "CHEMICAL":  # TODO: handle multi-class case!
                        entities.append((ent.start_char, ent.end_char, ent.text))
            return list(set(entities))
        
        elif self.model_type == "huggingface":
            tokenized_sentence = self.tokenizer.tokenize(sent)
            num_tokens = len(tokenized_sentence)

            # If the number of tokens exceeds the limit, split into chunks
            max_chunk_size = 505  # Keeping space for [CLS] and [SEP] tokens

            if num_tokens > max_chunk_size:
                print(f"Number of tokens ({num_tokens}) too large, splitting into chunks {sent[:50] + '[...]'}.")
                # Split into chunks of size max_chunk_size
                chunks = [tokenized_sentence[i:i + max_chunk_size] for i in range(0, num_tokens, max_chunk_size)]
                combined_results = []
                offset = 0  # To keep track of character offset due to chunks

                for chunk in chunks:
                    # Convert the chunk back into a string for NER
                    chunk_sent = self.tokenizer.convert_tokens_to_string(chunk)
                    
                    # Perform NER on the chunk
                    ner_results = self.nlp(chunk_sent)
                    
                    # Adjust the start and end char positions by the current offset
                    for entity in ner_results:
                        entity['start'] += offset
                        entity['end'] += offset
                        combined_results.append(entity)
                    
                    # Update offset by the length of the chunk (token to char conversion length)
                    offset += len(chunk_sent)
                
                # Optionally group and normalize the results
                if self.use_custom_entities_grouping:
                    combined_results = self.group_entities_custom_for_biobert(combined_results)
                
                if self.normalize_pred_representation:
                    return self.normalize_representation(combined_results)
                else:
                    return combined_results

            else:
                # If the number of tokens is within the limit, proceed normally
                ner_results = self.nlp(sent)
                if self.use_custom_entities_grouping:
                    ner_results = self.group_entities_custom_for_biobert(ner_results)

                if self.normalize_pred_representation:
                    return self.normalize_representation(ner_results)
                else:
                    return ner_results
        elif self.model_type == "regex":
            tokens_cleaned = tokenized_sent
            return find_drugs_and_conditions_normalized_BIO_output(tokens_cleaned)  # returns a tuple drug_matches, {"entites":all_char_indices}
        else:
            raise ValueError("Wrong model type. Allowed values are spacy and huggingface.")

    def normalize_representation(self, ner_results_combined):
        results = []
        for ent_dict in ner_results_combined:
            if self.entity_class_names_dict:
                entity_class_full_name = self.entity_class_names_dict[ent_dict['entity_group']]
            else:
                entity_class_full_name = ent_dict['entity_group']
            results.append((ent_dict['start'], ent_dict['end'], entity_class_full_name, ent_dict['word']))
        return results
    
    def group_entities_custom(self, entities, tokenized_sentence):
        """
        Groups entities based on specific rules.

        This function takes a list of entity dictionaries and groups them if:
        - The current entity starts a new group (entity type starts with 'B-' or in the case of an IO schema, differs from the current group).
        - The start of the next entity matches the end of the current entity, and they are of the same type.

        Parameters:
            entities (list): A list of dictionaries where each dictionary represents an entity with the following keys:
                - 'entity': The type of the entity (e.g., 'B-PER', 'I-PER', 'PER' in IO schema).
                - 'score': Confidence score of the entity.
                - 'index': Index of the token.
                - 'word': The word or sub-word token.
                - 'start': Start character index of the entity.
                - 'end': End character index of the entity.

        Returns:
            list: A list of grouped entities where each grouped entity is represented as a dictionary with:
                - 'entity_group': The type of the entity group.
                - 'score': The minimum confidence score within the group.
                - 'index': Index of the first token in the group.
                - 'word': The concatenated words of the group.
                - 'start': Start character index of the group.
                - 'end': End character index of the group.
        """
        grouped_entities = []
        current_entity = None

        for entity in entities:
            entity_type = entity['entity'][2:] if entity['entity'].startswith('B-') or entity['entity'].startswith('I-') else entity['entity']
            entity_start = entity['start']
            
            # case of the same word, should be grouped together
            if current_entity and current_entity['end'] == entity_start:
                if "##" in entity['word']:
                    next_token = entity['word'].replace('##', '')
                else:
                    next_token = " " + entity['word']
                current_entity['word'] += next_token
                current_entity['end'] = entity['end']
                current_entity['score'] = min(current_entity['score'], entity['score'])

            # Check if the entity starts a new group
            elif not current_entity or (entity_start -1) != current_entity['end'] or entity_type != current_entity['entity_group']:
                if current_entity:
                    grouped_entities.append(current_entity)
                entity_word = entity['word']
                if entity_word.startswith("##"):
                    # begin entities should not start with ## -> there is a tokenizer issue?? TODO: deeper understanding why this is happening
                    for reduce_index_by in range(2,6):
                        if entity['index'] >= reduce_index_by:
                            previous_token = tokenized_sentence[entity['index']-reduce_index_by]
                            entity_word = previous_token + entity_word.replace('##', '')
                            if not(entity_word.startswith("##")):
                                break
                if entity_word.startswith("-"):
                    # begin entities should not start with - -> there is a tokenizer issue?? TODO: deeper understanding why this is happening
                    for reduce_index_by in range(2,6):
                        if entity['index'] >= reduce_index_by:
                            previous_token = tokenized_sentence[entity['index']-reduce_index_by]
                            entity_word = previous_token + entity_word.replace('##', '')
                            if not(entity_word.startswith("-")):
                                break
                if entity_word.startswith("]") or entity_word.startswith(")"):
                    # begin entities should not start with ] or ) -> there is a tokenizer issue?? TODO: deeper understanding why this is happening
                    for reduce_index_by in range(2,6):
                        if entity['index'] >= reduce_index_by:
                            previous_token = tokenized_sentence[entity['index']-reduce_index_by]
                            entity_word = previous_token + entity_word
                            if entity_word.startswith("(") or entity_word.startswith("["):
                                break             
                current_entity = {
                    'entity_group': entity_type,
                    'score': entity['score'],
                    'index': entity['index'],
                    'word': entity_word,
                    'start': entity_start,
                    'end': entity['end']
                }
            else:
                # Extend the current entity group if the start matches the previous end and types match
                if "##" in entity['word']:
                    next_token = entity['word'].replace('##', '')
                else:
                    next_token = " " + entity['word']
                current_entity['word'] += next_token
                current_entity['end'] = entity['end']
                current_entity['score'] = min(current_entity['score'], entity['score'])

        # Append the last entity if it exists
        if current_entity:
            grouped_entities.append(current_entity)

        return grouped_entities
    
        

    def group_entities_custom_for_biobert(self, entities):
        grouped_entities = []
        current_entity = None

        for entity in entities:
            if entity['entity'].startswith('B-') and ("##" not in entity['word']):
                if current_entity:
                    grouped_entities.append(current_entity)
                current_entity = {
                    'entity_group': entity['entity'][2:],
                    'score': entity['score'],
                    'index': entity['index'],
                    'word': entity['word'].replace('##', ''),
                    'start': entity['start'],
                    'end': entity['end']
                }
            elif current_entity:
                if "##" in entity['word']:
                    next_token = entity['word'].replace('##', '')
                else:
                    next_token = " " + entity['word']
                current_entity['word'] += next_token
                current_entity['end'] = entity['end']
                current_entity['score'] = min(current_entity['score'], entity['score'])

        if current_entity:
            grouped_entities.append(current_entity)

        return grouped_entities

    def combine_entity_subwords(self, sent, entities):

        result = []
        current_entity = None
        merged_dict = {}
        if not entities:
            return []

        for entity in entities:
            if entity["entity"].startswith("B") or entity["entity"].startswith("I"):
                if current_entity is not None and entity["entity"].startswith("I"):
                    if entity['word'].startswith('##'):
                        current_entity['word'] += entity['word'][2:]
                    elif current_entity['end'] == entity['start']:
                        current_entity['word'] += entity['word']  # no space needed
                    else:
                        current_entity['word'] += " "
                        current_entity['word'] += entity['word']
                    current_entity['end'] = entity['end']
                    current_entity['score'] = round((current_entity['score'] + entity['score']) / 2,
                                                    3)  # average of the confidence
                else:
                    current_entity = entity.copy()
                    current_entity['entity'] = entity['entity'].replace("B-", "").replace("I-",
                                                                                          "")  # TODO: does the I- make sense?
                    if self.entity_class_names_dict:
                        current_entity['entity'] = self.entity_class_names_dict[current_entity['entity']]
                    current_entity['word'] = entity['word'][2:] if entity['word'].startswith('##') else entity['word']
                    result.append(current_entity)

        if self.return_words_only:
            current_entity_start = 0
            for i, entity in enumerate(result):

                if entity['entity'] == 'B':
                    merged_dict[i] = entity['word']
                    current_entity_start = i
                elif entity['entity'] == 'I':
                    if merged_dict:
                        merged_dict[current_entity_start] += ' ' + entity['word']
                    else:
                        print("could not save entity: {} from sentence {}. All entities found: {}".format(entity, sent,
                                                                                                          entities))

            return list(merged_dict.values())
        return result
