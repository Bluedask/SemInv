import os
import numpy as np
import torch
from datasets import Dataset
import transformers
import re
import warnings
import json
import random
import math
from tqdm import trange, tqdm
import argparse
from sklearn.metrics import confusion_matrix
from utils.json import JsonAdapter
import traceback
from itertools import product

warnings.filterwarnings("ignore")

class ConditionTest():
    def __init__(self, seed):
        random.seed(seed)
    
    def condition_exec(self, troj_model_dir, gpt_tokenizer_filepath, gpt_embedding_filepath, 
            bert_tokenizer_filepath, bert_embedding_filepath, output_dir, troj_model_id, inverse_data_dir
        ):
        # load trigger from file
        trigger_path = os.path.join(inverse_data_dir, f'{troj_model_id}.log')
        def process(sentence):
            word_list = []
            for word in sentence.split(' '):
                if word == '':
                    continue
                if word[0] in ['Ġ']:
                    word = word[1:]
                word_list.append(word)
            return ' '.join(word_list)

        with open(trigger_path, 'r') as f:
            text = f.readlines()
            re_list = []
            count = 0
            for line in text:
                if '[Scanning Result]' in line:
                    trigger = re.findall(r'trigger: (.*?)  loss:', line)
                    loss = re.findall(r'loss: (\d+\.\d+)', line)
                    if trigger != []:
                        trigger = trigger[0]
                        trigger = process(trigger)
                        loss = float(loss[0])
                        re_list.append([trigger, loss, count])
                        re_list = sorted(re_list, key=lambda x: x[1], reverse=False)
                        count += 1

        source_labels = [0, 1]
        target_labels = [0, 1]
        spatial_ratios = ['random', 'start', 'end']
        combinations = list(product(source_labels, target_labels, spatial_ratios, re_list))
        for combination in tqdm(combinations):
            source_label, target_label, spatial_ratio, trigger = combination
            output_path = f'{output_dir}/{spatial_ratio}_{source_label}_{target_label}_t{trigger[2]}.json'
            self.exec(troj_model_dir, gpt_tokenizer_filepath, gpt_embedding_filepath, 
            bert_tokenizer_filepath, bert_embedding_filepath, output_path, troj_model_id, spatial_ratio, trigger, source_label, target_label)
        
    def exec(self, troj_model_dir, gpt_tokenizer_filepath, gpt_embedding_filepath, 
            bert_tokenizer_filepath, bert_embedding_filepath, output_path, 
            troj_model_id, spatial_ratio, ori_trigger, source_class, target_class
        ):
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        trigger_text = ori_trigger[0]
        trigger_loss = ori_trigger[1]
        trigger = trigger_text, source_class, target_class
        res_dict = {}

        # ----------------------------------------------------------------------------------
        # ===: Get input parameters
        # ----------------------------------------------------------------------------------
        troj_config_path = f'{troj_model_dir}/config.json'
        model_filepath = f'{troj_model_dir}/model.pt'
        with open(troj_config_path, 'r') as f:
            troj_config = json.load(f)
        cls_token_is_first = troj_config['cls_token_is_first']
        num_classes = troj_config['number_classes']
        
        # ----------------------------------------------------------------------------------
        # ===: Load model
        # ----------------------------------------------------------------------------------
        target_model = torch.load(model_filepath).to(device)
        if cls_token_is_first:
            tokenizer = transformers.DistilBertTokenizer.from_pretrained(bert_tokenizer_filepath)
            embedding_model = torch.load(bert_embedding_filepath).to(device)
        else:
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(gpt_tokenizer_filepath)
            embedding_model = torch.load(gpt_embedding_filepath).to(device)
            tokenizer.pad_token = tokenizer.eos_token
        max_length = embedding_model.config.max_position_embeddings

        # ----------------------------------------------------------------------------------
        # ===: Load Dataset
        # ----------------------------------------------------------------------------------
        sentences = []
        labels = []
        sort_key = lambda x: int(re.findall(r'\d+', x)[0])*100+int(re.findall(r'\d+', x)[1])
        clean_example_dir = f'{troj_model_dir}/clean_example_data'
        clean_example_path_list = sorted(os.listdir(clean_example_dir), key=sort_key)
        for clean_data_id in clean_example_path_list:
            with open(os.path.join(clean_example_dir, clean_data_id)) as f:
                clean_data = f.read()
                sentences.append(clean_data)
                labels.append(int(re.findall(r'\d+', clean_data_id)[0]))
        troj_dataset = Dataset.from_dict({'sentence': sentences, 'label': labels})
        
        # ----------------------------------------------------------------------------------
        # ===: Poison Dataset
        # ----------------------------------------------------------------------------------
        poison_sens, poison_labels, poison_locs, poison_filename_list, len_list = \
            self.dataset_poison(troj_dataset, trigger, clean_example_path_list, spatial_ratio)
        poison_encodings = tokenizer(poison_sens, max_length=max_length, truncation=True)

        # ----------------------------------------------------------------------------------
        # ===: Compute asr
        # ----------------------------------------------------------------------------------
        asr, poison_logits = self.test(
            target_model=target_model, 
            embedding_model=embedding_model, 
            encodings=poison_encodings, 
            labels=poison_labels, 
            device=device,
            cls_token_is_first=cls_token_is_first,
            troj_model_id=troj_model_id 
        )
        poison_preds = [int(np.argmax(value)) for value in poison_logits]
        cm = confusion_matrix(poison_labels, poison_preds, labels=range(num_classes))

        res_dict['trigger'] = {
            'text': trigger[0],
            'loss': trigger_loss,
            'source_label': trigger[1],
            'target_label': trigger[2]
        }
        res_dict['attack_success_rate'] = asr
        res_dict['confusion_matric'] = {
            'TP': int(cm[1, 1]),
            'FN': int(cm[1, 0]),
            'FP': int(cm[0, 1]),
            'TN': int(cm[0, 0])
        }
        poison_res = {}
        for i in range(len(poison_labels)):
            poison_res[poison_filename_list[i]] = {
                'poison_success': True if poison_labels[i] == poison_preds[i] else False, 
                'poison_label': poison_labels[i],
                'poison_pred': poison_preds[i], 
                'poison_logits': poison_logits[i].tolist(),
                'poison_location': poison_locs[i],
                'sentence_length': len_list[i],
            }
        res_dict['poison_res'] = poison_res
        with open(output_path, 'w') as f:
            json.dump(res_dict, f, indent=4)


    def generate(self, target_model, embedding_model, input_ids, attention_mask, cls_token_is_first, device):
        target_model.eval()
        embedding_model.eval()

        with torch.no_grad():
            ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            attention_tensor = torch.LongTensor(attention_mask).unsqueeze(0).to(device)

            if cls_token_is_first:
                ids_embedding = embedding_model.get_input_embeddings()(ids_tensor)

                position_ids = torch.arange(ids_embedding.shape[1], dtype=torch.long).to(device)
                position_ids = position_ids.unsqueeze(0).expand([ids_embedding.shape[0], ids_embedding.shape[1]])
                position_embedding = embedding_model.embeddings.position_embeddings(position_ids)

                norm_sen_embedding = ids_embedding + position_embedding
                norm_sen_embedding = embedding_model.embeddings.LayerNorm(norm_sen_embedding)
                norm_sen_embedding = embedding_model.embeddings.dropout(norm_sen_embedding)

                embedding_vector = embedding_model(inputs_embeds=norm_sen_embedding, attention_mask=attention_tensor)[0]
                embedding_vector = embedding_vector[:, 0, :].unsqueeze(1)
            else:
                ids_embedding = embedding_model.get_input_embeddings()(ids_tensor)
                embedding_vector = embedding_model(inputs_embeds=ids_embedding, attention_mask=attention_tensor)[0]
                embedding_vector = embedding_vector[:, -1, :].unsqueeze(1)
            logits = target_model(embedding_vector).cpu().detach().numpy().squeeze()
            pred = np.argmax(logits)
        return pred, logits

    def poison(self, clean_sen, trigger_text, spatial_ratio='random'):
        clean_sen_list = clean_sen.split(' ')
        sen_length = len(clean_sen_list)
     
        # ----------------------------------------------------------------------------------
        # ===: Insert random location except for spatial condition
        # ----------------------------------------------------------------------------------
        if spatial_ratio == 'random':
            insert_loc = random.randint(0, sen_length)
        elif spatial_ratio == 'start':
            insert_loc = 0
        elif spatial_ratio == 'end':
            insert_loc = sen_length
        else:
            spatial_ratio = float(spatial_ratio)
            insert_loc = int(spatial_ratio * sen_length)
        trigger_text = trigger_text.split(' ')
        troj_sen_list = clean_sen_list[:insert_loc] + trigger_text + clean_sen_list[insert_loc:]
        return ' '.join(troj_sen_list), round(insert_loc / sen_length, 2), sen_length

    def dataset_poison(self, clean_sen_list, trigger, clean_example_path_list, spatial_ratio='random'):
        trigger_text, source_class, target_class = trigger
        poison_sen_list = []
        poison_label_list = []
        poison_loc_list = []
        poison_filename_list = []
        len_list = []

        sorted_sen_list = []
        for i in range(len(clean_sen_list)):
            clean_sen = clean_sen_list[i]
            if clean_sen['label'] == source_class:
                sorted_sen_list.append(clean_sen_list[i])
                poison_filename_list.append(clean_example_path_list[i]) 
        clean_sen_list = sorted_sen_list

        for clean_sen in clean_sen_list:
            poison_sen, poison_loc, sen_length = self.poison(clean_sen['sentence'], trigger_text, spatial_ratio)
            poison_sen_list.append(poison_sen)
            poison_label_list.append(target_class)
            poison_loc_list.append(poison_loc)
            len_list.append(sen_length)
        return poison_sen_list, poison_label_list, poison_loc_list, poison_filename_list, len_list

    def test(self, target_model, embedding_model, encodings, labels, cls_token_is_first, device, troj_model_id=0):
        target_model.eval()
        embedding_model.eval()
        acc_list = []
        logits_list = []
        for idx in range(len(labels)):
            input_ids = encodings['input_ids'][idx]
            attention_mask = encodings['attention_mask'][idx]
            label = labels[idx]

            with torch.no_grad():
                ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
                attention_tensor = torch.LongTensor(attention_mask).unsqueeze(0).to(device)
                if cls_token_is_first:
                    ids_embedding = embedding_model.get_input_embeddings()(ids_tensor)

                    position_ids = torch.arange(ids_embedding.shape[1], dtype=torch.long).to(device)
                    position_ids = position_ids.unsqueeze(0).expand([ids_embedding.shape[0], ids_embedding.shape[1]])
                    position_embedding = embedding_model.embeddings.position_embeddings(position_ids)

                    norm_sen_embedding = ids_embedding + position_embedding
                    norm_sen_embedding = embedding_model.embeddings.LayerNorm(norm_sen_embedding)
                    norm_sen_embedding = embedding_model.embeddings.dropout(norm_sen_embedding)

                    embedding_vector = embedding_model(inputs_embeds=norm_sen_embedding, attention_mask=attention_tensor)[0]
                    embedding_vector = embedding_vector[:, 0, :].unsqueeze(1)
                else:
                    ids_embedding = embedding_model.get_input_embeddings()(ids_tensor)
                    embedding_vector = embedding_model(inputs_embeds=ids_embedding, attention_mask=attention_tensor)[0]
                    embedding_vector = embedding_vector[:, -1, :].unsqueeze(1)
                logits = target_model(embedding_vector).cpu().detach().numpy().squeeze()
            pred = np.argmax(logits)
            acc_list.append(pred==label)
            logits_list.append(logits)

        return np.average(acc_list), logits_list


if __name__ == '__main__':

    output_default = 'output/condition/sem/spatial'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../../data', type=str)
    parser.add_argument('--troj_round', default='round6-train-dataset', type=str)
    parser.add_argument('--troj_model_id', default='id-00000000', type=str)
    parser.add_argument('--gpt_tokenizer_filepath', default='../../data/huggingface_models/gpt2', type=str)
    parser.add_argument('--bert_tokenizer_filepath', default='../../data/huggingface_models/distilbert-base-uncased', type=str)
    parser.add_argument('--output_root', default=output_default)
    parser.add_argument('--seed', default=1234, type=int)

    # move trigger info into inner modudle
    # parser.add_argument('--trigger_text', default='[PAD]', type=str)
    # parser.add_argument('--source_label', default=0, type=int)
    # parser.add_argument('--target_label', default=1, type=int)
    # parser.add_argument('--spatial_ratio', default='random', type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    
    # ----------------------------------------------------------------------------------
    # ===: parameter setting for spatial testing
    # ----------------------------------------------------------------------------------


    args = parser.parse_args()
    troj_data_dir = os.path.join(args.data_root, args.troj_round)
    troj_model_dir = os.path.join(troj_data_dir, 'models', args.troj_model_id)
    gpt_embedding_filepath = os.path.join(troj_data_dir, 'embeddings/GPT-2-gpt2.pt')
    bert_embedding_filepath = os.path.join(troj_data_dir, 'embeddings/DistilBERT-distilbert-base-uncased.pt')

    gpt_tokenizer_filepath = args.gpt_tokenizer_filepath
    bert_tokenizer_filepath = args.bert_tokenizer_filepath
    gpu_id = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    troj_model_id = args.troj_model_id
    output_dir = f'{args.output_root}/{args.troj_round}/{troj_model_id}'

    inverse_data_dir = f'scratch/sem/{args.troj_round}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    poison_tester = ConditionTest(args.seed)
    try:
        poison_tester.condition_exec(
            troj_model_id=troj_model_id,
            troj_model_dir=troj_model_dir,
            gpt_embedding_filepath=gpt_embedding_filepath,
            bert_embedding_filepath=bert_embedding_filepath,
            gpt_tokenizer_filepath=gpt_tokenizer_filepath,
            bert_tokenizer_filepath=bert_tokenizer_filepath,
            output_dir=output_dir, 
            inverse_data_dir=inverse_data_dir
        )
    except Exception as e:
        traceback.print_exc()
        print(f'{troj_model_id} occurs error!')