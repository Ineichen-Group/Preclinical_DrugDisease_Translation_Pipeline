import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics_vs_threshold(entity_type, thresholds: np.ndarray, f1_scores: list, precisions: list, recalls: list) -> None:
    """
    Plot F1 Score, Precision, and Recall versus cdist Threshold.

    Parameters:
    thresholds (np.ndarray): Array of threshold values.
    f1_scores (list): List of F1 scores corresponding to the thresholds.
    precisions (list): List of precision scores corresponding to the thresholds.
    recalls (list): List of recall scores corresponding to the thresholds.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(thresholds, f1_scores, label='F1 Score',  marker='x', color= "#DC267F", linewidth=2.5)
    plt.plot(thresholds, precisions, label='Precision', color= "#648FFF", linewidth=2.5)
    plt.plot(thresholds, recalls, label='Recall', color="#FE6100", linewidth=2.5)

    # Find the index of the highest F1 score
    highest_f1_idx = np.argmax(f1_scores)
    highest_f1_value = f1_scores[highest_f1_idx]
    highest_f1_threshold = thresholds[highest_f1_idx]

    # Highlight the highest F1 score
    plt.scatter(highest_f1_threshold, highest_f1_value, s=150, edgecolors='black', facecolors='none', linewidths=2, 
                label=f'Highest F1: {highest_f1_value:.2f} (cdist={highest_f1_threshold:.2f})')


    plt.xlabel('Embeddings distance (cdist) threshold value', fontsize=19)
    plt.title(f'Entity Linking Performance for {entity_type}', fontsize=19)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc='lower right')
    plt.grid(linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"04_normalization/viz/linking_performance_at_cdist_thresholds_{entity_type}.png")
    
    plt.show()


def update_termid_based_on_cdist(df, predicted_id_col, predicted_cdist_col, cdist_threshold):
    # Extract the relevant columns as tuples
    termid_cdist_tuples = list(zip(df[predicted_id_col], df[predicted_cdist_col]))
    
    # Apply the threshold to filter and update the snomed_termid
    updated_termid = ['n.a.' if cdist > cdist_threshold else termid for termid, cdist in termid_cdist_tuples]
    
    return updated_termid

def clean_and_convert(ids):
    # If ids is a list, convert it to a pandas Series
    if isinstance(ids, list):
        ids = pd.Series(ids)
    elif isinstance(ids, np.ndarray):
        ids = pd.Series(ids.tolist())
    
    # Replace 'n.a.' with '0'
    ids = ids.replace('n.a.', '0')
    
    # Convert all entries to integers, using 0 where conversion fails
    return pd.to_numeric(ids, errors='coerce').fillna(0).astype(int)

def calculate_linking_precision_recall_f1(correct_entity_ids, true_text, predicted_entity_ids, predicted_text, y_pred_cdist):
    correctly_linked = 0
    mentions_should_be_linked = 0
    mentions_linked_by_system_total = 0
    wrong_predictions = []
    missing_predictions = []

    for correct, predicted, true_txt, predicted_txt, cdist in zip(correct_entity_ids, predicted_entity_ids, true_text, predicted_text, y_pred_cdist):
        if correct != '0':
            mentions_should_be_linked += 1
            if predicted == '0':
                missing_predictions.append((correct, true_txt, predicted, predicted_txt, cdist))
        if predicted != '0':
            mentions_linked_by_system_total += 1
            if predicted == correct:
                correctly_linked += 1
            else:
                wrong_predictions.append((correct, true_txt, predicted, predicted_txt, cdist))
    
    # Calculate Precision
    precision = correctly_linked / mentions_linked_by_system_total if mentions_linked_by_system_total > 0 else 0
    
    # Calculate Recall
    recall = correctly_linked / mentions_should_be_linked if mentions_should_be_linked > 0 else 0
    
    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    performance_dict = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    return performance_dict, wrong_predictions, missing_predictions

def find_best_threshold(df: pd.DataFrame, source_col:str, target_id_col: str, predicted_id_col: str, predicted_text_col: str, predicted_cdist_col: str, thresholds: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Find the best threshold for updating the predicted_id_col based on evaluation metrics.

    Parameters:
    df (pd.DataFrame): The input DataFrame with necessary columns.
    thresholds (np.ndarray): Array of threshold values to evaluate.

    Returns:
    Tuple[float, Dict[str, Any]]: The best threshold and corresponding performance metrics.
    """
    y_true = df[target_id_col].to_numpy()
    y_true_mapped = np.array([0 if val == 'n.a.' else val for val in y_true])
    y_true_text = df[source_col].to_list()
    
    y_pred_text = df[predicted_text_col].to_list()
    y_pred_cdist = df[predicted_cdist_col].to_list()

    # Initialize lists to store performance metrics
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    wrong_predictions_at_best = []
    missing_predictions_at_best = []

    conf_matrices = []

    best_threshold = None
    best_f1_score = -1  # Initialize with a very low value

    for threshold in thresholds:
        # Update the snomed_termid values based on the cdist threshold
        updated_y_pred = update_termid_based_on_cdist(df, predicted_id_col, predicted_cdist_col, threshold)
 
        #y_pred_mapped = clean_and_convert(updated_y_pred)
        y_pred_mapped = np.array([0 if val == 'n.a.' else val for val in updated_y_pred])

        # Evaluate and store the metrics
        performance, wrong_predictions, missing_predictions = calculate_linking_precision_recall_f1(y_true_mapped, y_true_text, y_pred_mapped, y_pred_text, y_pred_cdist) #calculate_precision_recall_f1(y_true_mapped, y_pred_mapped)
        f1_scores.append(performance['f1_score'])
        precisions.append(performance['precision'])
        recalls.append(performance['recall'])

        # Update the best threshold based on F1 score
        if performance['f1_score'] > best_f1_score:
            best_f1_score = performance['f1_score']
            best_threshold = threshold
            wrong_predictions_at_best = wrong_predictions
            missing_predictions_at_best = missing_predictions

    # Return the best threshold and corresponding metrics
    best_performance = {
        'f1_score': best_f1_score,
        #'accuracy': accuracies[f1_scores.index(best_f1_score)],
        'precision': precisions[f1_scores.index(best_f1_score)],
        'recall': recalls[f1_scores.index(best_f1_score)],
        #'confusion_matrix': conf_matrices[f1_scores.index(best_f1_score)]
    }

    return best_threshold, best_performance, f1_scores, precisions, recalls, wrong_predictions_at_best, missing_predictions_at_best

def save_overview_performance(entity_type, thresholds, f1_scores, precisions, recalls):
    performance_df = pd.DataFrame({
        "threshold": thresholds,
        "f1_score": f1_scores,
        "precision": precisions,
        "recall": recalls})
    
    performance_df[["threshold", "f1_score", "precision", "recall"]] = performance_df[["threshold","f1_score", "precision", "recall"]].round(3)

    # Mark max F1
    performance_df["is_max_f1"] = performance_df["f1_score"] == performance_df["f1_score"].max()

    # Mark max precision, excluding 1.0
    precision_mask = performance_df["precision"] < 1.0
    max_precision = performance_df.loc[precision_mask, "precision"].max()
    performance_df["is_max_precision"] = performance_df["precision"] == max_precision

    # Mark max recall, excluding 1.0
    recall_mask = performance_df["recall"] < 1.0
    max_recall = performance_df.loc[recall_mask, "recall"].max()
    performance_df["is_max_recall"] = performance_df["recall"] == max_recall
    performance_df.to_csv(f"04_normalization/nen_stats/{entity_type}_cdist_threshold_performance.csv", index=False)

def main(entity_type, file_path):
    df_annotations = pd.read_csv(file_path)
    
    if entity_type == "disease":
        source_col = "linkbert_mapped_conditions"
        target_id_col = "mondo_target_id"
        predicted_id_col = "mondo_termid"
        predicted_cdist_col = "mondo_cdist"
        predicted_text_col = "mondo_term_norm"
    else:
        source_col = "linkbert_mapped_drugs"
        target_id_col = "umls_target_id"
        predicted_id_col = "umls_termid"
        predicted_cdist_col = "umls_cdist"
        predicted_text_col = "linkbert_umls_drugs"
    
    thresholds = np.arange(0, 15.0001, 0.05)

    best_threshold, best_performance, f1_scores, precisions, recalls, wrong_predictions_at_best, missing_preds_at_best = find_best_threshold(df_annotations, source_col, target_id_col, predicted_id_col, predicted_text_col, predicted_cdist_col, thresholds)
    save_overview_performance(entity_type, thresholds, f1_scores, precisions, recalls)
    
    wrong_preds_df = pd.DataFrame(wrong_predictions_at_best, columns=["true_id", "true_txt", "predicted_id", "predicted_txt", "cdist"])
    wrong_preds_df.to_csv(f"04_normalization/nen_stats/{entity_type}_cdist_{round(best_threshold,2)}_errors.csv", index=False)
    
    missing_preds_df = pd.DataFrame(missing_preds_at_best, columns=["true_id", "true_txt", "predicted_id", "predicted_txt", "cdist"])
    missing_preds_df.to_csv(f"04_normalization/nen_stats/{entity_type}_cdist_{round(best_threshold,2)}_missing_predictions.csv", index=False)
    
    plot_metrics_vs_threshold(entity_type, thresholds, f1_scores, precisions, recalls)
    
if __name__ == "__main__":
    entity_type = "disease"
    disease_annot_path = "04_normalization/data/mapped_to_embeddings_ontologies/sampled_conditions_manual_map_mondo_pred.csv"
    main(entity_type, disease_annot_path)
    
    entity_type = "drug"
    drug_annot_path = "04_normalization/data/mapped_to_embeddings_ontologies/sampled_drugs_manual_map_150_umls_pred.csv"
    main(entity_type, drug_annot_path)

