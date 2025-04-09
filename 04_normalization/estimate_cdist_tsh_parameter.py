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
    plt.legend(fontsize=18)
    plt.grid(linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"04_normalization/viz/linking_performance_at_cdist_thresholds_{entity_type}.pdf")
    
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

def calculate_linking_precision_recall_f1(correct_entity_ids, predicted_entity_ids):
    # Exclude positions where predicted_entity_ids == 0
    correctly_linked = sum(1 for correct, predicted in zip(correct_entity_ids, predicted_entity_ids) if predicted != 0 and correct == predicted)
    
    # Count all correct_entity_ids which are not 0
    mentions_should_be_linked = sum(1 for correct in correct_entity_ids if correct != 0)
    
    # Count all predicted_entity_ids which are not 0
    mentions_linked_by_system_total = sum(1 for predicted in predicted_entity_ids if predicted != 0)
    
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
    
    return performance_dict

def find_best_threshold(df: pd.DataFrame, target_id_col: str, predicted_id_col: str, predicted_cdist_col: str, thresholds: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Find the best threshold for updating snomed_termid based on evaluation metrics.

    Parameters:
    df (pd.DataFrame): The input DataFrame with necessary columns.
    thresholds (np.ndarray): Array of threshold values to evaluate.

    Returns:
    Tuple[float, Dict[str, Any]]: The best threshold and corresponding performance metrics.
    """
    y_true = df[target_id_col].to_numpy()
    y_true_mapped = np.array([0 if val == 'n.a.' else val for val in y_true])
    #y_true_mapped = clean_and_convert(df[target_id_col])

    # Initialize lists to store performance metrics
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []

    conf_matrices = []

    best_threshold = None
    best_f1_score = -1  # Initialize with a very low value

    for threshold in thresholds:
        # Update the snomed_termid values based on the cdist threshold
        updated_y_pred = update_termid_based_on_cdist(df, predicted_id_col, predicted_cdist_col, threshold)
 
        #y_pred_mapped = clean_and_convert(updated_y_pred)
        y_pred_mapped = np.array([0 if val == 'n.a.' else val for val in updated_y_pred])

        # Evaluate and store the metrics
        performance = calculate_linking_precision_recall_f1(y_true_mapped, y_pred_mapped) #calculate_precision_recall_f1(y_true_mapped, y_pred_mapped)
        f1_scores.append(performance['f1_score'])
        precisions.append(performance['precision'])
        recalls.append(performance['recall'])

        # Update the best threshold based on F1 score
        if performance['f1_score'] > best_f1_score:
            best_f1_score = performance['f1_score']
            best_threshold = threshold

    # Return the best threshold and corresponding metrics
    best_performance = {
        'f1_score': best_f1_score,
        #'accuracy': accuracies[f1_scores.index(best_f1_score)],
        'precision': precisions[f1_scores.index(best_f1_score)],
        'recall': recalls[f1_scores.index(best_f1_score)],
        #'confusion_matrix': conf_matrices[f1_scores.index(best_f1_score)]
    }

    return best_threshold, best_performance, f1_scores, precisions, recalls

if __name__ == "__main__":
    df_disease = pd.read_csv("04_normalization/data/mapped_to_embeddings_ontologies/sampled_conditions_manual_map_mondo_pred.csv")
    entity_type = "disease"
    target_id_col = "mondo_target_id"
    predicted_id_col = "mondo_termid"
    predicted_cdist_col = "mondo_cdist"
    thresholds = np.linspace(0, 15, 100)
    print(len(thresholds))
    best_threshold, best_performance, f1_scores, precisions, recalls = find_best_threshold(df_disease, target_id_col, predicted_id_col, predicted_cdist_col, thresholds)
    plot_metrics_vs_threshold(entity_type, thresholds, f1_scores, precisions, recalls)