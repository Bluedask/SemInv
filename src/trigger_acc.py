import re
import os
import csv
import json
import logging
from typing import List, Tuple, Dict, Any

# --- Configuration ---
BASE_DATA_PATH = '../../data'  # Base directory for datasets
# DATA_ROUND_NAME = 'round6-holdout-dataset' # Example, can be changed
DATA_ROUND_NAME = 'round6-train-dataset' # Using a common example for testing
INVERSE_LOG_DIR_BASE = 'scratch' # Base directory for inversion logs

# Log parsing constants
LOG_SCAN_RESULT_MARKER = '[Scanning Result]' # Was [Scanning Result]
# Regex to capture trigger and loss. Note the non-standard space ( ) before loss.
# If this space is a regular space in your logs, change '  loss:' to ' loss:'.
TRIGGER_REGEX = r"trigger: (.*?) \u00A0loss:" # Using unicode for the no-break space
LOSS_REGEX = r"loss: (\d+\.\d+)"

# Token processing constants
TOKEN_PREFIX_TO_STRIP = 'Ġ'

# Evaluation constants
NUM_TOP_TRIGGERS_FOR_AVG_ACC = 4

# Setup basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def normalize_trigger_tokens(tokens: List[str]) -> List[str]:
    """
    Normalizes a list of trigger tokens by removing specific prefixes and lowercasing.
    """
    processed_tokens = []
    for token in tokens:
        if token.startswith(TOKEN_PREFIX_TO_STRIP):
            token = token[len(TOKEN_PREFIX_TO_STRIP):]
        # Also remove leading/trailing spaces that might result from splitting or prefix stripping
        token = token.strip()
        if token: # Keep only non-empty tokens
            processed_tokens.append(token.lower())
    return processed_tokens

def parse_trigger_string(trigger_string: str) -> List[str]:
    """
    Splits a trigger string into tokens and normalizes them.
    """
    # Split by space, then filter out empty strings that may arise from multiple spaces
    raw_tokens = [token for token in trigger_string.split(' ') if token]
    return normalize_trigger_tokens(raw_tokens)

def calculate_trigger_similarity(predicted_tokens: List[str], true_tokens: List[str]) -> float:
    """
    Calculates the similarity of a predicted trigger to the true trigger.
    Metric: (number of common tokens) / (number of tokens in true trigger)
    """
    if not true_tokens:
        return 0.0  # Avoid division by zero and handle empty true trigger case
    
    common_tokens = sum(1 for token in predicted_tokens if token in true_tokens)
    # To be more precise like the original, count how many of the *predicted* tokens are in the *true* set.
    # The original logic was `sum(word in true_trigger_words for word in inverse_trigger_words)`
    # which is correct if `inverse_trigger_words` maps to `predicted_tokens`.
    
    # A slightly more robust version might be to count unique common tokens
    # common_unique_tokens = len(set(predicted_tokens) & set(true_tokens))
    # For now, sticking to the original logic's spirit:
    # number of predicted tokens that are present in the true_tokens list
    
    # Original logic: count how many predicted_tokens are also in true_tokens
    # This can exceed 1.0 if predicted_tokens is longer and has many common words or if words repeat.
    # The original code was: `sum(word in true_trigger_words for word in inverse_trigger_words) / len(true_trigger_words)`
    # This means "for each word in the predicted trigger, if it's in the true trigger, count it".
    # This is essentially a recall of true trigger words if predicted_tokens are unique and a subset.
    # If predicted_tokens can have duplicates, and true_tokens has duplicates, this can be tricky.
    # Let's assume tokens within a trigger are unique for this metric or that order doesn't matter
    # and we are checking for presence.

    # A more standard approach for set similarity like Jaccard would be:
    # intersection = len(set(predicted_tokens) & set(true_tokens))
    # union = len(set(predicted_tokens) | set(true_tokens))
    # similarity = intersection / union if union > 0 else 0.0

    # Sticking to the spirit of the original simple "accuracy":
    # Count how many of the *true_tokens* are found in the *predicted_tokens* set for recall-like behavior
    # Or, count how many of the *predicted_tokens* are found in the *true_tokens* set for precision-like behavior
    # The original was `sum(word_pred in true_words_set for word_pred in pred_words_list) / len(true_words_list)`
    
    # Let's refine to: fraction of true trigger words correctly found in the predicted trigger.
    # This makes the denominator fixed as len(true_tokens).
    # And numerator is number of unique true_tokens found in predicted_tokens.
    if not true_tokens:
        return 0.0
    
    true_tokens_set = set(true_tokens)
    predicted_tokens_set = set(predicted_tokens) # Consider unique predicted tokens
    
    correctly_found_true_tokens = len(true_tokens_set.intersection(predicted_tokens_set))
    
    return correctly_found_true_tokens / len(true_tokens_set)


# --- Data Loading and Processing Functions ---

def load_inversed_triggers_from_logs(inverse_log_directory: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Parses log files to extract inversed triggers and their corresponding losses.
    """
    inversed_trigger_data = {}
    if not os.path.isdir(inverse_log_directory):
        logger.error(f"Inverse log directory not found: {inverse_log_directory}")
        return inversed_trigger_data

    for log_filename in sorted(os.listdir(inverse_log_directory)):
        if not log_filename.endswith(".log"): # Assuming log files end with .log
            continue
        model_id = log_filename.split('.')[0]
        log_file_path = os.path.join(inverse_log_directory, log_filename)
        
        extracted_triggers_losses = []
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if LOG_SCAN_RESULT_MARKER in line:
                        trigger_match = re.findall(TRIGGER_REGEX, line)
                        loss_match = re.findall(LOSS_REGEX, line)
                        
                        if trigger_match and loss_match:
                            try:
                                trigger_text = trigger_match[0].strip()
                                loss_value = float(loss_match[0])
                                if trigger_text: # Ensure trigger is not empty
                                    extracted_triggers_losses.append((trigger_text, loss_value))
                            except (IndexError, ValueError) as e:
                                logger.warning(f"Could not parse trigger/loss from line in {log_file_path}: {line.strip()}. Error: {e}")
                        # else:
                            # logger.debug(f"No trigger/loss match in line: {line.strip()} from {log_file_path}")
        except FileNotFoundError:
            logger.warning(f"Log file not found: {log_file_path}")
        except Exception as e:
            logger.error(f"Error reading log file {log_file_path}: {e}")
            
        if extracted_triggers_losses:
            inversed_trigger_data[model_id] = {'inverse_trigger_candidates': extracted_triggers_losses}
        # else:
            # logger.info(f"No scanning results found for model_id: {model_id} in {log_file_path}")

    return inversed_trigger_data


def enrich_with_ground_truth(
    all_trigger_data: Dict[str, Dict[str, Any]],
    dataset_path: str,
    data_round_name: str
) -> Dict[str, Dict[str, Any]]:
    """
    Enriches the trigger data with ground truth information (true trigger, poisoned status).
    Returns a dictionary containing only data for poisoned models.
    """
    models_info_path = os.path.join(dataset_path, data_round_name, 'models')
    if not os.path.isdir(models_info_path):
        logger.error(f"Models directory not found: {models_info_path}")
        return {}

    poisoned_models_data = {}
    for model_id in sorted(os.listdir(models_info_path)):
        model_specific_path = os.path.join(models_info_path, model_id)
        if not os.path.isdir(model_specific_path):
            continue

        ground_truth_file = os.path.join(model_specific_path, 'ground_truth.csv')
        config_file = os.path.join(model_specific_path, 'config.json')

        if not (os.path.exists(ground_truth_file) and os.path.exists(config_file)):
            logger.warning(f"Ground truth or config file missing for model_id: {model_id}")
            continue
        
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as gt_f:
                # Assumes CSV has one row, one column for the label
                is_model_marked_poisoned_in_gt = list(csv.reader(gt_f))[0][0] 
            
            with open(config_file, 'r', encoding='utf-8') as cfg_f:
                model_config = json.load(cfg_f)
            
            is_poisoned_in_config = model_config.get('poisoned', False)
            
            # Process only if model_id was found in logs and config confirms poisoning
            if model_id in all_trigger_data and is_poisoned_in_config:
                current_model_data = all_trigger_data[model_id]
                current_model_data['true_trigger_text'] = model_config['triggers'][0]['text'] # Assumes one trigger
                current_model_data['is_poisoned_ground_truth'] = int(is_model_marked_poisoned_in_gt) # Store the label from CSV
                poisoned_models_data[model_id] = current_model_data

        except FileNotFoundError:
            logger.warning(f"File not found during ground truth processing for model_id: {model_id}")
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON config for model_id: {model_id}")
        except (IndexError, KeyError) as e:
            logger.warning(f"Data missing in config/ground_truth for model_id {model_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing ground truth for model_id {model_id}: {e}")
            
    return poisoned_models_data

# --- Main Script Logic ---

def main():
    """
    Main function to execute the trigger evaluation script.
    """
    # Construct paths
    inverse_log_directory = os.path.join(INVERSE_LOG_DIR_BASE, DATA_ROUND_NAME)
    
    # 1. Load inversed triggers from logs
    logger.info(f"Loading inversed triggers from: {inverse_log_directory}")
    all_models_inversed_data = load_inversed_triggers_from_logs(inverse_log_directory)
    if not all_models_inversed_data:
        logger.error("No inversed trigger data loaded. Exiting.")
        return
    logger.info(f"Loaded inversed trigger data for {len(all_models_inversed_data)} models.")

    # 2. Enrich with ground truth and filter for poisoned models
    logger.info(f"Loading ground truth for dataset: {DATA_ROUND_NAME}")
    final_poisoned_data_map = enrich_with_ground_truth(all_models_inversed_data, BASE_DATA_PATH, DATA_ROUND_NAME)
    if not final_poisoned_data_map:
        logger.error("No poisoned model data to evaluate after loading ground truth. Exiting.")
        return
    logger.info(f"Processing {len(final_poisoned_data_map)} confirmed poisoned models.")

    # 3. Calculate similarity scores for each poisoned model
    for model_id, data_item in final_poisoned_data_map.items():
        true_trigger_text = data_item.get('true_trigger_text')
        if not true_trigger_text:
            logger.warning(f"Model {model_id} is missing 'true_trigger_text'. Skipping accuracy calculation.")
            data_item['similarity_scores'] = []
            continue

        true_trigger_tokens = parse_trigger_string(true_trigger_text)
        if not true_trigger_tokens:
            logger.warning(f"Parsed true trigger for Model {model_id} is empty. Setting similarity to 0.")
            data_item['similarity_scores'] = [0.0] * len(data_item.get('inverse_trigger_candidates', []))
            continue
            
        # Sort inversed triggers by loss (ascending)
        # inverse_trigger_candidates is a list of (trigger_text, loss_value)
        sorted_inversed_triggers = sorted(
            data_item.get('inverse_trigger_candidates', []),
            key=lambda x: x[1],  # Sort by loss
            reverse=False
        )
        
        similarity_scores = []
        for inversed_trigger_text, loss_val in sorted_inversed_triggers:
            inversed_trigger_tokens = parse_trigger_string(inversed_trigger_text)
            similarity = calculate_trigger_similarity(inversed_trigger_tokens, true_trigger_tokens)
            similarity_scores.append(similarity)
        
        data_item['similarity_scores'] = similarity_scores
        # logger.info(f"Model: {model_id}, True: {true_trigger_tokens}, Best Inversed: {parse_trigger_string(sorted_inversed_triggers[0][0]) if sorted_inversed_triggers else 'N/A'}, Best Sim: {similarity_scores[0] if similarity_scores else 'N/A'}")

    # 4. Calculate and print average similarity for top N triggers
    average_similarity_top_n = []
    num_models_with_scores = 0
    
    # Determine the number of models that actually have similarity scores
    models_with_valid_scores = [
        data for data in final_poisoned_data_map.values() if data.get('similarity_scores')
    ]
    num_models_with_scores = len(models_with_valid_scores)

    if num_models_with_scores == 0:
        logger.warning("No models have similarity scores to average.")
        return

    for i in range(NUM_TOP_TRIGGERS_FOR_AVG_ACC):
        current_rank_total_similarity = 0.0
        valid_models_for_this_rank = 0
        for data_item in models_with_valid_scores:
            if i < len(data_item['similarity_scores']):
                current_rank_total_similarity += data_item['similarity_scores'][i]
                valid_models_for_this_rank += 1
        
        if valid_models_for_this_rank > 0:
            avg_sim_for_rank_i = current_rank_total_similarity / valid_models_for_this_rank
        else:
            # This case means no model had enough (i.e., i-th) trigger candidates
            avg_sim_for_rank_i = 0.0 
            # logger.info(f"No models had a {i+1}-th trigger candidate to average.")
        
        average_similarity_top_n.append(avg_sim_for_rank_i)
        logger.info(f"Average similarity for top {i+1} trigger candidate(s) (across {valid_models_for_this_rank} models): {avg_sim_for_rank_i:.4f}")

    logger.info(f"Final average similarities for top {NUM_TOP_TRIGGERS_FOR_AVG_ACC} candidates: {average_similarity_top_n}")
    # The script would implicitly make `average_similarity_top_n` available if run in an interactive session.
    # For standalone script, you might want to save this to a file or print more formally.
    print(f"Average Similarities for Top {NUM_TOP_TRIGGERS_FOR_AVG_ACC} Triggers: {average_similarity_top_n}")


if __name__ == "__main__":
    # Example: Change DATA_ROUND_NAME if you want to run for a different round
    # DATA_ROUND_NAME = 'roundX-some-dataset' 
    main()