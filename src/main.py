import argparse
import json
import logging
import os
import random
# import sys # Not directly used in this snippet, can be removed if logger doesn't need it
# import threading # Not directly used
# import time # Not directly used
# from datetime import datetime # Not directly used, Add_logger might use it

import numpy as np
import torch
import transformers # Keep as some specific tokenizers might be used if AutoTokenizer logic changes
import yaml
from pynvml import (nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo, nvmlInit, nvmlShutdown) # Specific imports

from torch.nn import CrossEntropyLoss
from tqdm import tqdm # trange is not used
# from transformers import (AutoModelForCausalLM, # Used in load_models
#                           AutoModelForSequenceClassification, # Not explicitly used, target_model is loaded directly
#                           AutoTokenizer) # Not explicitly used, specific tokenizers are loaded
from transformers import AutoModelForCausalLM, GPT2Tokenizer, DistilBertTokenizer

# Assuming these are custom utility modules/classes
# from utils import Add_utils, utils # Old way
# from utils.Add_logger import Logger # Old way

# New assumed structure for utils:
# from project_utils.logging_config import setup_experiment_logger # Example
# from project_utils.model_helpers import parse_architecture_from_tokenizer_path # Example

# For this refactor, I'll assume Logger and arch_parser are available simply as:
from utils.logger import Logger # Assuming this path is correct relative to the script
import utils.tool as project_arch_parser # Assuming utils.utils has arch_parser

# --- Configuration Constants (to be overridden by args or a main config file) ---
DEFAULT_SHARED_RESOURCES_DIR = './resources' # Placeholder for embeddings, tokenizers etc.
DEFAULT_REFERENCE_MODELS_DIR = './reference_models' # Placeholder for benign models


class TriggerReconstructionPipeline:
    def __init__(self, target_model_directory, output_base_dir,
                 hyperparameters_path, trigger_length_config,
                 random_seed, shared_resources_dir=None, reference_models_dir=None):

        self.shared_resources_dir = shared_resources_dir or DEFAULT_SHARED_RESOURCES_DIR
        self.reference_models_dir = reference_models_dir or DEFAULT_REFERENCE_MODELS_DIR

        self.target_model_dir = target_model_directory
        self.configured_trigger_length = trigger_length_config
        self.current_seed = random_seed

        # GPU Setup
        gpu_available, selected_gpu_id = self._configure_gpu(required_gpus=1, max_usage_threshold=0.05)
        if not gpu_available:
            raise EnvironmentError(f"No suitable GPU available. Attempted to use: {selected_gpu_id}")
        os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu_id
        self.active_gpu_id = selected_gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths
        self.target_model_path = os.path.join(self.target_model_dir, 'model.pt')
        self.clean_examples_path = os.path.join(self.target_model_dir, 'clean_example_data')
        self.model_identifier = os.path.basename(os.path.normpath(self.target_model_dir)) # More robust way to get id

        # Load Hyperparameters
        try:
            with open(hyperparameters_path) as f:
                self.hyperparams = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Hyperparameter file not found: {hyperparameters_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML hyperparameter file: {e}")


        # Load Target Model Configuration
        try:
            with open(os.path.join(self.target_model_dir, 'config.json')) as f:
                target_model_meta = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model config.json not found in {self.target_model_dir}")

        self.is_target_model_poisoned = target_model_meta.get('poisoned', False) # Default to False if key missing
        embedding_flavor = target_model_meta.get('embedding_flavor', 'distilbert') # Default flavor

        if embedding_flavor == 'gpt2':
            self.embedding_model_path = os.path.join(self.shared_resources_dir, 'embeddings', 'GPT-2-gpt2.pt')
            tokenizer_archive_path = os.path.join(self.shared_resources_dir, 'tokenizers', 'GPT-2-gpt2.pt')
            self.huggingface_model_name = 'gpt2'
        elif embedding_flavor == 'distilbert':
            self.embedding_model_path = os.path.join(self.shared_resources_dir, 'embeddings', 'DistilBERT-distilbert-base-uncased.pt')
            tokenizer_archive_path = os.path.join(self.shared_resources_dir, 'tokenizers', 'DistilBERT-distilbert-base-uncased.pt')
            self.huggingface_model_name = 'distilbert-base-uncased'
        else:
            raise ValueError(f"Unsupported embedding_flavor: {embedding_flavor}")

        self.model_architecture_name = project_arch_parser.arch_parser(tokenizer_archive_path) # was utils.arch_parser

        # Logger Setup
        # output_base_dir was scratch_dirpath
        os.makedirs(output_base_dir, exist_ok=True)
        log_file_path = os.path.join(output_base_dir, f"{self.model_identifier}_detection.log")
        self.experiment_logger = Logger(log_file_path, logging.ERROR, logging.DEBUG) # Assuming Logger is a class

        # Initialize state for individual inversion runs (will be reset by _initialize_inversion_run_settings)
        self._initialize_inversion_run_settings()


    def _configure_gpu(self, required_gpus=1, max_usage_threshold=0.5):
        if required_gpus == 0:
            return True, 'CPU_MODE' # Using a distinct string for CPU mode
        nvmlInit()
        num_gpus = nvmlDeviceGetCount()
        available_gpu_indices = []
        for i in reversed(range(num_gpus)): # Check higher indexed GPUs first
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            current_usage_percentage = mem_info.used / mem_info.total if mem_info.total > 0 else 1.0
            if current_usage_percentage < max_usage_threshold:
                available_gpu_indices.append(str(i))

        selected_gpus = available_gpu_indices[:required_gpus]
        nvmlShutdown()

        if len(selected_gpus) < required_gpus:
            return False, f"Found {len(selected_gpus)} suitable GPUs, but {required_gpus} required. Threshold: <{max_usage_threshold*100}% used."
        return True, ','.join(selected_gpus)

    def _initialize_inversion_run_settings(self, current_tokenizer=None, embedding_provider_model=None):
        """Sets hyperparameters for a single trigger inversion run and resets related state."""
        config = self.hyperparams # was self.config
        self.current_temperature = config['init_temp']
        self.max_temperature_value = config['max_temp']
        self.temperature_update_interval_epochs = config['temp_scaling_check_epoch']
        self.temperature_decrease_factor = config['temp_scaling_down_multiplier']
        self.temperature_increase_factor = config['temp_scaling_up_multiplier'] # was temp_scaling_down_multiplier in one case, likely a typo
        self.inversion_loss_threshold = config['loss_barrier']
        self.gradient_noise_level = config['noise_ratio']
        self.max_rollback_attempts = config['rollback_thres']

        self.num_inversion_epochs = config['epochs']
        self.optimizer_learning_rate = config['lr']
        self.scheduler_step_interval = config['scheduler_step_size']
        self.scheduler_decay_rate = config['scheduler_gamma']

        self.max_sequence_length = config['max_len']
        self.discreteness_epsilon = config['eps_to_one_hot'] # For checking closeness to one-hot

        # State for the current inversion run
        self.is_temperature_scaling_active = False
        self.current_rollback_count = 0
        self.current_run_best_asr = 0.0
        self.current_run_best_loss = float('inf')
        self.current_run_best_trigger_text = "N/A" # Initial value

        if current_tokenizer and embedding_provider_model:
            self.trigger_placeholder_token_id = current_tokenizer.pad_token_id
            self.trigger_placeholder_tokens = torch.ones(self.configured_trigger_length, device=self.device).long() * self.trigger_placeholder_token_id
            self.trigger_placeholder_attention_mask = torch.ones_like(self.trigger_placeholder_tokens)
            self.vocabulary_embeddings = embedding_provider_model.get_input_embeddings().weight.detach() # Detach to be safe


    def get_target_model_metadata(self):
        # Calculate model and embedding file sizes
        model_file_bytes = os.path.getsize(self.target_model_path) if os.path.exists(self.target_model_path) else 0
        embedding_file_bytes = os.path.getsize(self.embedding_model_path) if os.path.exists(self.embedding_model_path) else 0
        total_file_size_mb = (model_file_bytes + embedding_file_bytes) / (1024 * 1024)

        num_data_examples = 0
        if os.path.exists(self.clean_examples_path) and os.path.isdir(self.clean_examples_path):
            num_data_examples = len([name for name in os.listdir(self.clean_examples_path) if name.endswith('.txt')])

        model_status_label = "Poisoned Model" if self.is_target_model_poisoned else "Clean Model"

        return {
            'model_size_mb': f'{total_file_size_mb:.2f} MB',
            'num_clean_examples': num_data_examples,
            'model_architecture': self.model_architecture_name,
            'model_ground_truth': [model_status_label]
        }

    def run_full_detection(self):
        self._execute_detection_workflow()
        self.experiment_logger.info('Detection process completed.')

    def _execute_detection_workflow(self):
        self.experiment_logger.info('Starting detection workflow...')
        self.experiment_logger.info(f'Using GPU ID(s): {self.active_gpu_id}')
        self._set_all_random_seeds(self.current_seed)

        # Load models and tokenizer
        eval_model, embed_model, ref_benign_model, text_tokenizer = self._load_all_models_and_tokenizer()

        # Generate configurations to scan (victim label, target label, position)
        scan_configurations = self._generate_scan_configurations()

        overall_best_inversion_loss = float('inf')
        overall_best_result_details = {}
        all_scan_results = []

        for scan_config in tqdm(scan_configurations, desc="Scanning Configurations"):
            victim_class_idx = scan_config['victim_class']
            intended_class_idx = scan_config['intended_class']
            trigger_pos_setting = scan_config['position_setting']

            # Determine actual insertion index based on position setting
            if trigger_pos_setting == 'start':
                # `1` typically means after CLS token if present, or very beginning.
                actual_insert_idx = 1 # Or 0 if no CLS and inserting at absolute start
            elif trigger_pos_setting == 'end':
                actual_insert_idx = -1 # Sentinel for "end of sequence before padding"
            else: # e.g. 'middle' or specific index
                # This part needs more robust handling if other positions are supported.
                # For now, defaulting to a placeholder or raising error.
                self.experiment_logger.warning(f"Unsupported position: {trigger_pos_setting}, using start.")
                actual_insert_idx = 1


            self._initialize_inversion_run_settings(text_tokenizer, embed_model) # Reset for current scan

            clean_data_for_victim_class = self._load_data_for_class(victim_class_idx)
            input_token_ids, attention_masks = self._tokenize_text_data(text_tokenizer, clean_data_for_victim_class)
            
            # Insert placeholders for the trigger
            stamped_token_ids, stamped_attention_masks, actual_insertion_indices = \
                self._insert_trigger_placeholders(text_tokenizer, input_token_ids, attention_masks, actual_insert_idx)

            # Perform trigger inversion for the current configuration
            inverted_trigger_str, final_loss, final_asr = self._reconstruct_trigger_candidate(
                eval_model, embed_model, ref_benign_model, text_tokenizer,
                intended_class_idx, stamped_token_ids, stamped_attention_masks, actual_insertion_indices
            )

            current_scan_summary = {
                'victim_class': victim_class_idx,
                'intended_class': intended_class_idx,
                'position_setting': trigger_pos_setting,
                'reconstructed_trigger': inverted_trigger_str,
                'loss': final_loss,
                'asr': final_asr
            }
            all_scan_results.append(current_scan_summary)

            if final_loss < overall_best_inversion_loss:
                overall_best_inversion_loss = final_loss
                overall_best_result_details = current_scan_summary
        
        self.experiment_logger.info("--- All Scan Configuration Results ---")
        for res in all_scan_results:
            log_msg = (f"Victim Class: {res['victim_class']}, Target Class: {res['intended_class']}, "
                       f"Position: {res['position_setting']}, Loss: {res['loss']:.6f}, ASR: {res['asr']:.4f}, "
                       f"Trigger: '{res['reconstructed_trigger']}'")
            self.experiment_logger.result_collection(log_msg) # Assuming this logger method exists

        if overall_best_result_details:
            log_msg_best = (f"Overall Best Result -> Victim: {overall_best_result_details['victim_class']}, "
                            f"Target: {overall_best_result_details['intended_class']}, Pos: {overall_best_result_details['position_setting']}, "
                            f"Loss: {overall_best_result_details['loss']:.6f}, ASR: {overall_best_result_details['asr']:.4f}, "
                            f"Trigger: '{overall_best_result_details['reconstructed_trigger']}'")
            self.experiment_logger.best_result(log_msg_best) # Assuming this logger method exists
        else:
            self.experiment_logger.warning("No successful inversion results found.")


    def _reconstruct_trigger_candidate(self, model_to_attack, embedding_model_for_vocab,
                                     reference_clean_model, text_tokenizer,
                                     desired_output_class, input_ids_with_placeholders,
                                     attention_mask_for_placeholders, trigger_insertion_points):
        """
        Core logic for inverting (reconstructing) a trigger for a specific configuration.
        """
        # Optimizable tensor representing trigger token probabilities
        # vocab_size = embedding_model_for_vocab.get_input_embeddings().weight.shape[0] # Alternative
        vocab_size = text_tokenizer.vocab_size
        optimizable_trigger_logits = torch.zeros(self.configured_trigger_length, vocab_size,
                                               device=self.device, requires_grad=True)

        adam_optimizer = torch.optim.Adam([optimizable_trigger_logits], lr=self.optimizer_learning_rate)
        learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer,
                                                                  step_size=self.scheduler_step_interval,
                                                                  gamma=self.scheduler_decay_rate)
        
        # Initialize bests for this specific run (already done in _initialize_inversion_run_settings)

        for epoch_idx in range(self.num_inversion_epochs):
            # Forward pass through the model with the current continuous trigger representation
            output_logits_poisoned, output_logits_benign_ref = self._compute_model_outputs(
                epoch_idx, model_to_attack, embedding_model_for_vocab, reference_clean_model,
                adam_optimizer, optimizable_trigger_logits, # Pass the raw logits
                input_ids_with_placeholders, attention_mask_for_placeholders, trigger_insertion_points
            )

            # Calculate loss
            # Target labels should be shaped [batch_size]
            true_target_labels = torch.full((output_logits_poisoned.shape[0],), desired_output_class,
                                            dtype=torch.long, device=self.device)

            loss_function = CrossEntropyLoss()
            primary_task_loss = loss_function(output_logits_poisoned, true_target_labels)
            
            # Loss for benign model consistency (encourage trigger to NOT affect benign model, or affect it differently)
            # Assuming the benign model should predict the opposite class (1-target_label) or original class
            # This part might need adjustment based on the specific goal of benign_ce_loss
            benign_ref_target_labels = 1 - true_target_labels # Example: encourage opposite prediction
            benign_consistency_loss = loss_function(output_logits_benign_ref, benign_ref_target_labels)


            # Attack Success Rate (ASR)
            predicted_classes = torch.argmax(output_logits_poisoned, dim=1)
            successful_attacks = (predicted_classes == true_target_labels).sum().item()
            current_asr = successful_attacks / predicted_classes.shape[0]

            # Marginal benign loss penalty (from original code)
            if epoch_idx == 0:
                initial_benign_loss_bound = benign_consistency_loss.detach()
            
            penalized_benign_loss = torch.clamp(benign_consistency_loss - initial_benign_loss_bound, min=0)
            
            total_loss_for_optimizer = primary_task_loss + penalized_benign_loss # Add other regularizers if any

            # Backward pass and optimization step
            # Retain graph might be needed if parts of backbone_model are shared and also optimized,
            # or if multiple backward passes are needed for complex regularizers not shown here.
            # For simple CE loss on a detached backbone, it's often not needed.
            # The original code had it for distilbert. This usually means some part of the computation graph
            # needs to be preserved for a subsequent backward pass, or some parameters are not part of `opt_var`.
            # Given `opt_var` is the only thing optimized by `optimizer`, `retain_graph=True` might be for
            # second-order effects or complex interactions if `backbone_model` itself had `requires_grad=True` on some params.
            # For a standard trigger inversion, this is unusual if only `opt_var` is being tuned.
            # I'll keep it conditional as in original, assuming there was a reason.
            adam_optimizer.zero_grad() # Also zero grad on models if they were part of optim
            if self.model_architecture_name == 'distilbert':
                total_loss_for_optimizer.backward(retain_graph=True)
            else:
                total_loss_for_optimizer.backward()
            adam_optimizer.step()
            learning_rate_scheduler.step()

            self.last_primary_task_loss_in_epoch = primary_task_loss.item() # For temp scaling logic
            self.last_asr_in_epoch = current_asr # For temp scaling logic

            # Condition to start temperature scaling (adaptive learning phase)
            if primary_task_loss.item() <= self.inversion_loss_threshold:
                self.is_temperature_scaling_active = True
            
            # Check if current trigger representation is close to discrete (one-hot)
            # self.current_relaxed_trigger_probs is set in _compute_model_outputs
            top_probs, top_indices = torch.topk(self.current_relaxed_trigger_probs, 1, dim=1)
            # Difference from sum of top_probs to num_trigger_tokens (i.e., sum of 1s if one-hot)
            discreteness_metric = self.configured_trigger_length - torch.sum(top_probs).item()

            if discreteness_metric < self.discreteness_epsilon and primary_task_loss.item() <= self.inversion_loss_threshold:
                # Convert current best discrete trigger to text
                current_discrete_trigger_tokens = [text_tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in top_indices.squeeze()]
                self.current_run_best_trigger_text = " ".join(current_discrete_trigger_tokens)
                self.current_run_best_asr = current_asr
                self.current_run_best_loss = primary_task_loss.item()
                
                # Adjust loss barrier to seek even better solutions
                self.inversion_loss_threshold = self.current_run_best_loss / 2.0
                self.current_rollback_count = 0 # Reset rollbacks as we found a good discrete candidate

            # Update best loss found so far in this run (even if not fully discrete)
            if primary_task_loss.item() < self.current_run_best_loss: # Use primary task loss for this
                self.current_run_best_asr = current_asr # Update ASR along with best loss
                self.current_run_best_loss = primary_task_loss.item()
                # Potentially also save the top_indices here if we want the non-text version of the best continuous trigger
            
            log_message_epoch = (f"Epoch: {epoch_idx+1}/{self.num_inversion_epochs} | "
                                 f"Current Loss: {primary_task_loss.item():.4f} | Current ASR: {current_asr:.4f} | "
                                 f"Best Loss (this run): {self.current_run_best_loss:.4f} | Best ASR (this run): {self.current_run_best_asr:.4f}")
            self.experiment_logger.trigger_generation(log_message_epoch) # Assuming this logger method

        return self.current_run_best_trigger_text, self.current_run_best_loss, self.current_run_best_asr


    def _compute_model_outputs(self, current_epoch, model_to_evaluate, vocab_embedding_provider,
                             clean_reference_model, current_optimizer, # current_optimizer not used here, remove if not needed
                             continuous_trigger_logits, # was self.opt_var
                             base_input_ids, base_attention_masks,
                             insertion_locations):

        # Zero gradients for models if they are in eval mode but somehow grads were enabled
        # vocab_embedding_provider.zero_grad() # Typically in eval mode, grads shouldn't accumulate
        # model_to_evaluate.zero_grad()
        # clean_reference_model.zero_grad()

        # Temperature scaling and noise injection for Gumbel-Softmax / Softmax
        applied_noise = torch.zeros_like(continuous_trigger_logits, device=self.device)
        
        # Rollback logic (if stuck in poor local optima)
        if self.current_rollback_count >= self.max_rollback_attempts:
            self.current_rollback_count = 0
            # Increase loss barrier to relax constraints, ensure it's less than best loss found
            self.inversion_loss_threshold = min(self.inversion_loss_threshold * 2, self.current_run_best_loss - 1e-3 if self.current_run_best_loss < float('inf') else self.inversion_loss_threshold * 2)


        if current_epoch > 0 and (current_epoch % self.temperature_update_interval_epochs == 0):
            if self.is_temperature_scaling_active:
                if self.last_primary_task_loss_in_epoch < self.inversion_loss_threshold : # Check against the loss for this run
                    self.current_temperature /= self.temperature_decrease_factor
                else: # Loss is too high, try to escape
                    self.current_temperature *= self.temperature_increase_factor # Typo fix: was decrease factor
                    applied_noise = torch.rand_like(continuous_trigger_logits, device=self.device) * self.gradient_noise_level
                    if self.current_temperature > self.max_temperature_value:
                        self.current_temperature = self.max_temperature_value
                    self.current_rollback_count += 1
        
        # Apply temperature and softmax (or Gumbel-Softmax if that's the true intention)
        # The original code used `torch.softmax`, so `bound_opt_var` was probabilities.
        # Gumbel-Softmax is often used for sampling discrete tokens in a differentiable way.
        # If Gumbel-Softmax is intended, `torch.nn.functional.gumbel_softmax` should be used.
        # Assuming standard softmax as per original:
        self.current_relaxed_trigger_probs = torch.softmax(continuous_trigger_logits / self.current_temperature + applied_noise, dim=1)

        # Get actual trigger embeddings using these probabilities (soft embedding)
        # self.vocabulary_embeddings is detached
        current_trigger_word_embeddings = torch.tensordot(self.current_relaxed_trigger_probs,
                                                          self.vocabulary_embeddings,
                                                          dims=([1], [0]))
        
        # Get base sentence embeddings (without trigger)
        base_sentence_embeddings = vocab_embedding_provider.get_input_embeddings()(base_input_ids)

        # Batch-wise insertion of trigger embeddings
        final_input_embeddings_batch = []
        for i in range(base_input_ids.shape[0]):
            loc = insertion_locations[i].item() # Assuming scalar index per item
            
            # Ensure loc is within bounds to prevent slicing errors, considering trigger length
            # Max possible start for trigger is len(sequence) - trigger_len
            # If loc is too large, adjust or handle error.
            # This needs robust handling if loc can be near the end.
            # Example: loc = min(loc, base_sentence_embeddings.shape[1] - self.configured_trigger_length)
            
            part_before_trigger = base_sentence_embeddings[i, :loc, :]
            part_after_trigger = base_sentence_embeddings[i, loc + self.configured_trigger_length:, :] # Placeholder len

            # Ensure trigger_word_embeddings fits the batch item, might need squeeze if batch size is 1 for trigger
            # If trigger_word_embeddings is [TriggerLen, EmbDim]
            if current_trigger_word_embeddings.ndim == 2: # Single trigger for all batch items or specific logic
                 assembled_embeddings = torch.cat((part_before_trigger, current_trigger_word_embeddings, part_after_trigger), dim=0)
            elif current_trigger_word_embeddings.ndim == 3 and current_trigger_word_embeddings.shape[0] == base_input_ids.shape[0]: # Batch of triggers
                 assembled_embeddings = torch.cat((part_before_trigger, current_trigger_word_embeddings[i], part_after_trigger), dim=0)
            else: # Fallback or error for unexpected shapes
                 assembled_embeddings = torch.cat((part_before_trigger, current_trigger_word_embeddings.squeeze(), part_after_trigger), dim=0)

            final_input_embeddings_batch.append(assembled_embeddings)
        
        final_input_embeddings = torch.stack(final_input_embeddings_batch)

        # Model-specific forward pass details (e.g., position embeddings for DistilBERT)
        if self.model_architecture_name == 'distilbert':
            # For DistilBERT, position_ids are usually handled internally if not provided.
            # If custom position embeddings are needed due to manipulation:
            position_ids_tensor = torch.arange(self.max_sequence_length, dtype=torch.long, device=self.device)
            position_ids_tensor = position_ids_tensor.unsqueeze(0).expand_as(base_input_ids) # Expand to batch
            explicit_position_embeddings = vocab_embedding_provider.embeddings.position_embeddings(position_ids_tensor)
            
            embeddings_with_position = final_input_embeddings + explicit_position_embeddings
            normalized_embeddings = vocab_embedding_provider.embeddings.LayerNorm(embeddings_with_position)
            final_dropout_embeddings = vocab_embedding_provider.embeddings.dropout(normalized_embeddings)
            
            # Pass to the main body of the embedding provider (which acts as backbone here)
            # The output is typically [batch_size, seq_len, hidden_dim]
            backbone_output = vocab_embedding_provider(inputs_embeds=final_dropout_embeddings, attention_mask=base_attention_masks)[0]
            # For sequence classification, usually CLS token output is used
            pooled_output = backbone_output[:, 0, :].unsqueeze(1) # [batch_size, 1, hidden_dim]
        elif self.model_architecture_name == 'gpt2':
            # GPT-2 might not need explicit position embedding addition if inputs_embeds is used,
            # but custom handling might be needed if sequence structure is heavily modified.
            # The original code did not add position_embeddings for GPT2 here.
            backbone_output = vocab_embedding_provider(inputs_embeds=final_input_embeddings, attention_mask=base_attention_masks)[0]
             # For GPT-2 sequence classification, often the last token's hidden state is used.
            pooled_output = backbone_output[:, -1, :].unsqueeze(1) # [batch_size, 1, hidden_dim]
        else:
            raise NotImplementedError(f"Forward pass for {self.model_architecture_name} not fully implemented.")

        # Final classification layer (the 'target_model' and 'benign_model')
        logits_from_target = model_to_evaluate(pooled_output)
        logits_from_benign_ref = clean_reference_model(pooled_output)

        return logits_from_target, logits_from_benign_ref


    def _insert_trigger_placeholders(self, text_tokenizer, original_input_ids, original_attention_mask, insert_location_sentinel):
        """
        Inserts placeholder tokens for the trigger into the input sequences.
        insert_location_sentinel: 1 for start, -1 for end. Other specific indices could be supported.
        """
        batch_size, seq_len = original_input_ids.shape
        
        # Create tensors for stamped inputs
        stamped_ids = torch.full_like(original_input_ids, text_tokenizer.pad_token_id)
        stamped_mask = torch.zeros_like(original_attention_mask)
        
        actual_insertion_indices = []

        for i in range(batch_size):
            current_ids = original_input_ids[i]
            current_mask = original_attention_mask[i]
            
            # Find actual length of sequence before padding
            actual_seq_len = current_mask.sum().item()

            if insert_location_sentinel == 1: # Start of sequence (e.g., after CLS)
                # Assuming CLS is token 0 if present, insert at index 1.
                # If no CLS, insert at index 0. For simplicity, using 1.
                idx_to_insert = 1
            elif insert_location_sentinel == -1: # End of sequence
                # Insert before padding, or at max_len - trigger_len if sentence is full
                idx_to_insert = max(0, actual_seq_len - self.configured_trigger_length)
                # Ensure it doesn't go past max_len - trigger_len to allow space
                idx_to_insert = min(idx_to_insert, self.max_sequence_length - self.configured_trigger_length)

            else: # Specific index passed
                idx_to_insert = insert_location_sentinel
                # Clamp to be safe
                idx_to_insert = max(0, min(idx_to_insert, self.max_sequence_length - self.configured_trigger_length))


            actual_insertion_indices.append(idx_to_insert)

            # Assemble the new sequence with placeholders
            # Part 1: Before trigger
            stamped_ids[i, :idx_to_insert] = current_ids[:idx_to_insert]
            stamped_mask[i, :idx_to_insert] = current_mask[:idx_to_insert]
            
            # Part 2: Trigger placeholders
            placeholder_end = idx_to_insert + self.configured_trigger_length
            stamped_ids[i, idx_to_insert:placeholder_end] = self.trigger_placeholder_tokens
            stamped_mask[i, idx_to_insert:placeholder_end] = self.trigger_placeholder_attention_mask
            
            # Part 3: After trigger
            # Number of original tokens to shift from after placeholder original position
            num_original_tokens_after = self.max_sequence_length - placeholder_end
            original_tokens_to_copy_start = idx_to_insert # Original tokens that were at placeholder start
            
            stamped_ids[i, placeholder_end:] = current_ids[original_tokens_to_copy_start : original_tokens_to_copy_start + num_original_tokens_after]
            stamped_mask[i, placeholder_end:] = current_mask[original_tokens_to_copy_start : original_tokens_to_copy_start + num_original_tokens_after]

            # Special handling for GPT-2 if last token becomes pad after insertion (from original code)
            # This ensures the model has a non-pad token to attend to if sequence classification relies on last token.
            if self.model_architecture_name == 'gpt2':
                if placeholder_end < self.max_sequence_length and stamped_ids[i, self.max_sequence_length-1] == text_tokenizer.pad_token_id:
                    # Try to find the last valid token from original sequence BEFORE truncation by placeholder insertion
                    # This logic needs to be careful not to pick up a pad token if original was short.
                    if actual_seq_len > 0 : # If there was any content
                        last_original_meaningful_token_idx = actual_seq_len -1
                        # Ensure this index is valid for current_ids before insertion shift
                        if original_tokens_to_copy_start + (last_original_meaningful_token_idx - original_tokens_to_copy_start) < current_ids.size(0):
                           # This logic is a bit complex and error-prone. A simpler GPT-2 end-of-sequence handling might be needed.
                           # The original code copied `raw_input_ids[idx,last_valid_token_idx]`.
                           # `last_valid_token_idx` was `(raw_input_ids[idx] == tokenizer.pad_token_id).nonzero()[0] - 1`
                           # This should be based on `current_ids` and its `actual_seq_len`.
                           if current_ids[last_original_meaningful_token_idx] != text_tokenizer.pad_token_id:
                               stamped_ids[i, self.max_sequence_length-1] = current_ids[last_original_meaningful_token_idx]
                               stamped_mask[i, self.max_sequence_length-1] = 1


        return stamped_ids, stamped_mask, torch.tensor(actual_insertion_indices, device=self.device)


    def _tokenize_text_data(self, text_tokenizer, text_list_to_tokenize):
        tokenized_output = text_tokenizer(
            text_list_to_tokenize,
            max_length=self.max_sequence_length,
            padding='max_length', # Pad to max_length
            truncation=True,      # Truncate if longer
            return_tensors='pt'   # Return PyTorch tensors
        )
        return tokenized_output['input_ids'].to(self.device), tokenized_output['attention_mask'].to(self.device)

    def _generate_scan_configurations(self):
        # Define possible class labels (e.g., for binary classification)
        possible_class_labels = [0, 1]
        # Define trigger insertion strategies
        insertion_strategies = ['start', 'end'] # Could add 'middle' or specific indices

        scan_configs = []
        for victim_cls in possible_class_labels:
            for target_cls in possible_class_labels:
                if target_cls != victim_cls: # Ensure target is different from victim
                    for strategy in insertion_strategies:
                        config = {
                            'victim_class': victim_cls,
                            'intended_class': target_cls,
                            'position_setting': strategy
                        }
                        scan_configs.append(config)
        return scan_configs

    def _load_data_for_class(self, victim_class_id_filter=None):
        example_files = []
        if os.path.isdir(self.clean_examples_path):
            example_files = [os.path.join(self.clean_examples_path, f)
                               for f in os.listdir(self.clean_examples_path) if f.endswith('.txt')]
        example_files.sort() # Ensure consistent order

        loaded_text_data = []
        for file_path in example_files:
            # Assuming filename format like 'id_SOMEID_LABEL_SOMETHING.txt'
            # This parsing is fragile; a metadata file would be better.
            try:
                filename_parts = os.path.basename(file_path).split('_')
                # Example: ex_id-00000000_cls-0_idx-0.txt -> parts[-3] is cls-0 -> parts[-1] of that is 0
                file_class_label = int(filename_parts[-3].split('-')[-1]) # Adjust if format differs
            except (IndexError, ValueError):
                self.experiment_logger.warning(f"Could not parse class label from filename: {file_path}. Skipping.")
                continue

            if victim_class_id_filter is None or file_class_label == victim_class_id_filter:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_handle:
                        text_content = f_handle.read().strip()
                        if text_content: # Ensure non-empty
                             loaded_text_data.append(text_content)
                except Exception as e:
                    self.experiment_logger.error(f"Error reading file {file_path}: {e}")
        return loaded_text_data

    def _load_all_models_and_tokenizer(self):
        """Loads the target model, embedding/backbone model, a reference benign model, and tokenizer."""
        try:
            model_under_test = torch.load(self.target_model_path, map_location=self.device)
            # This is the "backbone" used for embeddings and potentially some transformer layers
            embedding_providing_model = torch.load(self.embedding_model_path, map_location=self.device)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading critical model files: {e}")


        # Determine tokenizer and benign model path based on architecture
        # Paths to Hugging Face local models should be configurable
        huggingface_local_repo_path = os.path.join(self.shared_resources_dir, 'huggingface_models', self.huggingface_model_name)

        if self.model_architecture_name == 'distilbert':
            text_tokenizer = DistilBertTokenizer.from_pretrained(huggingface_local_repo_path)
            # Path for reference benign model (needs to be robustly configurable)
            # Example: using a fixed ID from a reference set
            benign_model_filename = 'model_distilbert_benign_ref.pt' # Placeholder name
            benign_ref_model_path = os.path.join(self.reference_models_dir, self.model_architecture_name, benign_model_filename)
            # Original hardcoded: f'{DATA_DIR}/round6-train-dataset/models/id-00000006/model.pt'

        elif self.model_architecture_name == 'gpt2':
            # GPT-2 specific setup (e.g., for perplexity if 'sim_model' was for that)
            # self.perplexity_reference_model = AutoModelForCausalLM.from_pretrained(huggingface_local_repo_path, output_hidden_states=True).to(self.device)
            # self.perplexity_reference_model.eval()
            text_tokenizer = GPT2Tokenizer.from_pretrained(huggingface_local_repo_path)
            text_tokenizer.pad_token = text_tokenizer.eos_token # Common practice for GPT-2 padding
            benign_model_filename = 'model_gpt2_benign_ref.pt' # Placeholder name
            benign_ref_model_path = os.path.join(self.reference_models_dir, self.model_architecture_name, benign_model_filename)
            # Original hardcoded: f'{DATA_DIR}/round6-train-dataset/models/id-00000001/model.pt'
        else:
            raise NotImplementedError(f"Transformer architecture '{self.model_architecture_name}' not supported for model loading.")

        try:
            reference_benign_classifier = torch.load(benign_ref_model_path, map_location=self.device)
        except FileNotFoundError:
            self.experiment_logger.warning(f"Reference benign model not found at {benign_ref_model_path}. Proceeding without it if possible, or this might error later.")
            reference_benign_classifier = None # Allow running without if logic permits

        # Set models to evaluation mode
        model_under_test.eval()
        embedding_providing_model.eval()
        if reference_benign_classifier:
            reference_benign_classifier.eval()

        return model_under_test, embedding_providing_model, reference_benign_classifier, text_tokenizer

    def _set_all_random_seeds(self, seed_value):
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # For multi-GPU
        # Settings for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(description="Semantic Backdoor Inversion (SemInv) Tool")
    cli_parser.add_argument('--seed', default=1234, type=int, help="Random seed for reproducibility.")
    cli_parser.add_argument('--target_model_dir', required=True, type=str, help="Directory containing the target model.pt and config.json, and clean_example_data/.")
    cli_parser.add_argument('--output_dir_base', default='./experiment_outputs', type=str, help="Base directory to save logs and any other outputs.")
    cli_parser.add_argument('--hyperparams_file', required=True, type=str, help="Path to the YAML hyperparameter configuration file.")
    cli_parser.add_argument('--trigger_len', default=10, type=int, help="Length of the trigger to be inverted.")
    # Removed gpu_num as GPU selection is now more dynamic. Can add specific GPU ID selection if needed.
    cli_parser.add_argument('--shared_resources_path', default=DEFAULT_SHARED_RESOURCES_DIR, type=str, help="Path to shared resources like embeddings and tokenizers.")
    cli_parser.add_argument('--reference_models_path', default=DEFAULT_REFERENCE_MODELS_DIR, type=str, help="Path to reference models, like benign classifiers.")


    args = cli_parser.parse_args()

    # Construct a unique output directory for this run
    # Example: ./experiment_outputs/Seed1234/model_id_XYZ
    model_id_from_path = os.path.basename(os.path.normpath(args.target_model_dir))
    # round_arch = args.target_model_dir.split('/')[-3] # This is very specific to a fixed input path structure
    # A more robust way might be to get arch from model_config or a naming convention.
    # For now, let's use a simpler output path structure.
    current_run_output_dir = os.path.join(args.output_dir_base, f"Seed{args.seed}", model_id_from_path)
    os.makedirs(current_run_output_dir, exist_ok=True)


    try:
        pipeline_instance = TriggerReconstructionPipeline(
            target_model_directory=args.target_model_dir,
            output_base_dir=current_run_output_dir,
            hyperparameters_path=args.hyperparams_file,
            trigger_length_config=args.trigger_len,
            random_seed=args.seed,
            shared_resources_dir=args.shared_resources_path,
            reference_models_dir=args.reference_models_path
        )
        pipeline_instance.run_full_detection()
    except Exception as e:
        logging.error(f"Critical error during pipeline execution: {e}", exc_info=True) # Log traceback
        print(f"An error occurred. Check logs for details. Error: {e}")