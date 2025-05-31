import os
import re
import csv
import json
import logging
from typing import Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Configuration ---
# Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Keep if essential for your setup

# Data and Directory Config
DATA_ROUND_NAME = 'round6-test-dataset' # Example, change as needed
BASE_DATA_PATH = '../../data'
BASE_LOG_PATH = 'scratch' # Base for experiment log directories

# Log parsing
LOG_MARKER_BEST_ESTIMATION = '[Best Estimation]'
# Regex to capture trigger and loss. Note the non-standard space (  U+00A0) before loss.
# If this space is a regular space in your logs, change ' \u00A0loss:' to ' loss:'.
TRIGGER_REGEX_PATTERN = r"trigger: (.*?) \u00A0loss:"
LOSS_REGEX_PATTERN = r"loss: (\d+\.\d+)"

# Method names (used as prefixes for keys in data_dict)
METHOD_PRIMARY_NAME = 'semantic_method'
METHOD_SECONDARY_NAME = 'baseline_method'

# Thresholds for determining positive prediction based on loss
# (model_type: loss_threshold)
PREDICTION_THRESHOLDS = {
    METHOD_PRIMARY_NAME: {
        'gpt2': 0.1,
        'distilbert-base-uncased': 0.1,
    },
    METHOD_SECONDARY_NAME: {
        'gpt2': 0.1,
        'distilbert-base-uncased': 0.001,
    }
}

# Setup basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_ground_truth_and_config(base_models_dir: str) -> Dict[str, Dict[str, Any]]:
    """Loads ground truth labels and model configurations."""
    model_data_map = {}
    if not os.path.isdir(base_models_dir):
        logger.error(f"Models directory not found: {base_models_dir}")
        return model_data_map

    for model_id in sorted(os.listdir(base_models_dir)):
        model_specific_path = os.path.join(base_models_dir, model_id)
        if not os.path.isdir(model_specific_path):
            continue

        gt_file = os.path.join(model_specific_path, 'ground_truth.csv')
        cfg_file = os.path.join(model_specific_path, 'config.json')

        if not (os.path.exists(gt_file) and os.path.exists(cfg_file)):
            logger.warning(f"Ground truth or config file missing for model: {model_id}")
            continue
        
        try:
            with open(gt_file, 'r', encoding='utf-8') as f_gt:
                ground_truth_label = int(list(csv.reader(f_gt))[0][0])
            with open(cfg_file, 'r', encoding='utf-8') as f_cfg:
                model_config = json.load(f_cfg)

            data_entry = {
                'ground_truth_label': ground_truth_label,
                'embedding_type': model_config.get('embedding_flavor', 'unknown')
            }
            if ground_truth_label == 1 and 'triggers' in model_config and model_config['triggers']:
                data_entry['true_trigger_text'] = model_config['triggers'][0].get('text')
            
            model_data_map[model_id] = data_entry

        except FileNotFoundError:
            logger.warning(f"File missing during ground truth loading for model: {model_id}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing ground truth CSV for model {model_id}: {e}")
        except json.JSONDecodeError:
            logger.warning(f"Error decoding JSON config for model: {model_id}")
        except Exception as e:
            logger.error(f"Unexpected error loading data for model {model_id}: {e}")
            
    return model_data_map


def parse_experiment_logs_for_method(
    log_directory: str,
    existing_model_data_map: Dict[str, Dict[str, Any]],
    method_name_prefix: str
):
    """Parses experiment logs to extract trigger and loss for a given method."""
    if not os.path.isdir(log_directory):
        logger.warning(f"Log directory for {method_name_prefix} not found: {log_directory}")
        return

    for log_filename in sorted(os.listdir(log_directory)):
        if not log_filename.endswith(".log"): # Or other relevant extension
            continue
        model_id = log_filename.split('.')[0]

        if model_id not in existing_model_data_map:
            # logger.warning(f"Model ID {model_id} from logs not found in ground truth data. Skipping.")
            continue # Only process models we have ground truth for

        log_file_path = os.path.join(log_directory, log_filename)
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f_log:
                for line in f_log:
                    if LOG_MARKER_BEST_ESTIMATION in line:
                        trigger_match = re.findall(TRIGGER_REGEX_PATTERN, line)
                        loss_match = re.findall(LOSS_REGEX_PATTERN, line)

                        if trigger_match and loss_match:
                            loss_key = f"{method_name_prefix}_loss"
                            trigger_key = f"{method_name_prefix}_trigger_text"
                            existing_model_data_map[model_id][loss_key] = float(loss_match[0])
                            existing_model_data_map[model_id][trigger_key] = trigger_match[0].strip()
                            break # Assuming one best estimation per file
                        # else:
                            # logger.debug(f"No trigger/loss match in best estimation line: {line.strip()} (File: {log_filename})")
        except FileNotFoundError:
            logger.warning(f"Log file not found: {log_file_path}")
        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing trigger/loss in {log_file_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading log file {log_file_path}: {e}")


def get_method_prediction(
    model_entry: Dict[str, Any],
    method_name_prefix: str,
    threshold_config: Dict[str, Dict[str, float]]
) -> int:
    """Determines a prediction (0 or 1) based on a method's loss and thresholds."""
    loss_key = f"{method_name_prefix}_loss"
    model_embedding_type = model_entry.get('embedding_type')
    
    if loss_key not in model_entry or model_embedding_type not in threshold_config.get(method_name_prefix, {}):
        # logger.debug(f"Missing loss or threshold for {method_name_prefix}, model embedding {model_embedding_type}. Defaulting to predict 0.")
        return 0 # Default to negative if data or threshold is missing for this method

    method_loss = model_entry[loss_key]
    loss_threshold = threshold_config[method_name_prefix][model_embedding_type]

    return 1 if method_loss <= loss_threshold else 0


def print_evaluation_metrics(true_labels: List[int], predicted_labels: List[int], method_description: str):
    """Calculates and prints standard classification metrics."""
    if not true_labels or not predicted_labels or len(true_labels) != len(predicted_labels):
        logger.error(f"Cannot calculate metrics for {method_description}: Invalid input lists.")
        return
    
    logger.info(f"\n--- Evaluation Metrics for: {method_description} ---")
    try:
        logger.info(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
        # Ensure labels for classification_report are present in both true and pred
        # If only one class is predicted, classification_report might warn/error.
        # common_labels = sorted(list(set(true_labels) | set(predicted_labels)))
        report = classification_report(true_labels, predicted_labels, zero_division=0)
        logger.info(f"Classification Report:\n{report}")
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]) # Explicitly define labels
        logger.info(f"Confusion Matrix (Rows: True, Cols: Pred) for labels [0, 1]:\n{conf_matrix}")
    except Exception as e:
        logger.error(f"Error calculating metrics for {method_description}: {e}")
    print("-" * 50)


# --- Main Evaluation Logic ---

def main():
    """Main script execution."""
    models_base_dir = os.path.join(BASE_DATA_PATH, DATA_ROUND_NAME, 'models')
    
    # 1. Load ground truth and base model configurations
    all_model_data = load_ground_truth_and_config(models_base_dir)
    if not all_model_data:
        logger.info("No ground truth data loaded. Exiting.")
        return
    logger.info(f"Loaded ground truth for {len(all_model_data)} models.")

    # 2. Load results for Primary Method (e.g., Semantic)
    # Original sem_data_dir = f'scratch/char/{data_round}' - 'char' might be method-specific
    primary_method_log_dir = os.path.join(BASE_LOG_PATH, 'char', DATA_ROUND_NAME) # Adjust 'char' if it varies
    parse_experiment_logs_for_method(primary_method_log_dir, all_model_data, METHOD_PRIMARY_NAME)
    logger.info(f"Parsed logs for {METHOD_PRIMARY_NAME} from {primary_method_log_dir}")

    # 3. Load results for Secondary Method (e.g., Baseline)
    # Original _data_dir = f'scratch/{data_round}'
    secondary_method_log_dir = os.path.join(BASE_LOG_PATH, DATA_ROUND_NAME) # Simple directory for secondary
    parse_experiment_logs_for_method(secondary_method_log_dir, all_model_data, METHOD_SECONDARY_NAME)
    logger.info(f"Parsed logs for {METHOD_SECONDARY_NAME} from {secondary_method_log_dir}")
    
    # --- Evaluations ---
    
    # Strategy 1: Evaluate Primary Method Independently
    true_labels_all = []
    primary_method_preds = []
    for model_id, data_item in all_model_data.items():
        if 'ground_truth_label' not in data_item: continue # Skip if essential GT is missing
        true_labels_all.append(data_item['ground_truth_label'])
        primary_method_preds.append(get_method_prediction(data_item, METHOD_PRIMARY_NAME, PREDICTION_THRESHOLDS))
    print_evaluation_metrics(true_labels_all, primary_method_preds, f"Independent '{METHOD_PRIMARY_NAME}'")

    # Strategy 2: Evaluate Secondary Method Independently
    # Note: true_labels_all can be reused if the set of models is the same
    secondary_method_preds = []
    for model_id, data_item in all_model_data.items():
        if 'ground_truth_label' not in data_item: continue
        # true_labels_all is already populated
        secondary_method_preds.append(get_method_prediction(data_item, METHOD_SECONDARY_NAME, PREDICTION_THRESHOLDS))
    print_evaluation_metrics(true_labels_all, secondary_method_preds, f"Independent '{METHOD_SECONDARY_NAME}'")

    # Strategy 3: Evaluate Hybrid Method (Primary, fallback to Secondary)
    hybrid_preds = []
    # true_labels_all can be reused
    for model_id, data_item in all_model_data.items():
        if 'ground_truth_label' not in data_item: continue
        
        primary_pred = get_method_prediction(data_item, METHOD_PRIMARY_NAME, PREDICTION_THRESHOLDS)
        
        if primary_pred == 1: # If primary method predicts positive
            hybrid_preds.append(1)
        else: # Otherwise, use secondary method's prediction
            secondary_pred = get_method_prediction(data_item, METHOD_SECONDARY_NAME, PREDICTION_THRESHOLDS)
            hybrid_preds.append(secondary_pred)
            
    print_evaluation_metrics(true_labels_all, hybrid_preds, f"Hybrid ('{METHOD_PRIMARY_NAME}' fallback to '{METHOD_SECONDARY_NAME}')")

    # Original First Evaluation Block (Subset evaluation - kept for reference, but less ideal)
    # This block evaluated METHOD_SECONDARY_NAME only on the subset where METHOD_PRIMARY_NAME was positive.
    # This is generally not how you'd evaluate METHOD_SECONDARY_NAME independently.
    # The independent evaluation above (Strategy 2) is preferred for METHOD_SECONDARY_NAME.
    logger.info("\n--- Original First Evaluation Block (Subset for Secondary Method - For Reference) ---")
    original_block_true_labels = []
    original_block_primary_preds = [] # Predictions from primary method that passed its threshold
    original_block_secondary_preds_on_subset = [] # Secondary method's prediction ON THAT SUBSET

    for model_id, data_item in all_model_data.items():
        if 'ground_truth_label' not in data_item: continue
        
        # Check if primary method would predict positive
        primary_pred_for_subset = get_method_prediction(data_item, METHOD_PRIMARY_NAME, PREDICTION_THRESHOLDS)
        
        if primary_pred_for_subset == 1:
            original_block_true_labels.append(data_item['ground_truth_label'])
            original_block_primary_preds.append(1) # By definition, primary method predicted 1 here
            
            # Now, what would secondary method predict for THIS item?
            secondary_pred_for_this_item = get_method_prediction(data_item, METHOD_SECONDARY_NAME, PREDICTION_THRESHOLDS)
            original_block_secondary_preds_on_subset.append(secondary_pred_for_this_item)

    print_evaluation_metrics(original_block_true_labels, original_block_primary_preds, f"Original Logic: '{METHOD_PRIMARY_NAME}' (on subset where it's positive)")
    print_evaluation_metrics(original_block_true_labels, original_block_secondary_preds_on_subset, f"Original Logic: '{METHOD_SECONDARY_NAME}' (on subset where primary was positive)")


if __name__ == "__main__":
    main()