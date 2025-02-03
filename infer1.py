import json
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
import torch
from plot import plot
import argparse
import pandas as pd

from models.units import UniTS, UniTSArgs
from data.dataset import IMUDataset

def get_args_parser():
    parser = argparse.ArgumentParser('inference', add_help=False)
    # Model parameters
    parser.add_argument('-l', '--load_path', default=None, type=str,
                        help='path to load pretrained model')

    # Dataset parameters
    parser.add_argument('-d', '--data_config', nargs='+', default=None,
                        help='dataset config path')

    parser.add_argument('-o', '--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--enable_aware', action='store_true', help='enable aware layer')
    parser.add_argument('--enable_cross', action='store_true', help='evaluate cross-dataset')
    return parser

args = get_args_parser().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_path = args.load_path
save_path = args.output_dir
enable_aware = args.enable_aware
enable_cross = args.enable_cross

num_class = 7
config_paths = args.data_config

default_loc = ['upperarm', 'wrist', 'waist', 'thigh']

os.makedirs(save_path, exist_ok=True)

def eval(eval_file):
    import os
    import numpy as np
    import json
    from collections import OrderedDict
    from transformers import AutoTokenizer, BertModel
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_list = ['downstairs', 'jog', 'lie', 'sit', 'stand', 'upstairs', 'walk']
    bert_model = BertModel.from_pretrained('bert-large-uncased').to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', model_max_length=512)

    # Helper functions
    def get_bert_embedding(text):
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = bert_model(**inputs.to(device))
        return torch.mean(outputs.last_hidden_state[0], dim=0).cpu().detach().numpy()

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def classify_text(pred_text, label_dict):
        pred_text = pred_text.lower()
        if 'down' in pred_text:
            return 0, False
        elif 'jog' in pred_text:
            return 1, False
        elif 'lie' in pred_text:
            return 2, False
        elif 'sit' in pred_text:
            return 3, False
        elif 'stand' in pred_text:
            return 4, False
        elif 'up' in pred_text:
            return 5, False
        elif 'walk' in pred_text:
            return 6, False
        else:
            scores = [cosine_similarity(get_bert_embedding(pred_text), embed) for embed in label_dict.values()]
            return np.argmax(scores), True

    def preprocess_text(text):
        if len(text) >= 20:
            text = text[-20:]
        return text

    def preprocess_truth(truth):
        truth = truth.split('\n')[0].strip()
        if truth.startswith('.') and truth[1:].startswith('.'):
            truth = truth[3:]
        elif truth.startswith('Answer: '):
            truth = truth[11:]
        return truth

    def plot_confusion_matrix(cm, labels, save_path):
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(np.arange(len(labels)), labels, rotation=90)
        plt.yticks(np.arange(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        # Add label values to the confusion matrix cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        
        plt.savefig(save_path, dpi=300)
        plt.close()

    # Load and prepare label embeddings
    label_dict = OrderedDict((label, get_bert_embedding(label)) for label in label_list)
    
    out_of_list_embeddings = []
    error_list = []
    
    with open(eval_file, 'r') as f:
        data = json.load(f)

    num_sample = len(data)
    num_class = len(label_list)
    all_pred = np.zeros(num_sample, dtype=int)
    all_truth = np.zeros(num_sample, dtype=int)
    
    for idx, item in enumerate(data):
        truth_text = preprocess_truth(item['ref'])
        pred_text = preprocess_text(item['pred'])
        pred_text = item['pred']
        truth_idx = label_list.index(truth_text)
        pred_idx, is_out = classify_text(pred_text, label_dict)
        if is_out:
            item['pred_label'] = label_list[pred_idx]
            out_of_list_embeddings.append(item)
        all_truth[idx] = truth_idx
        all_pred[idx] = pred_idx
        if truth_idx != pred_idx:
            error_list.append(item)

    # Metrics and saving results
    acc = accuracy_score(all_truth, all_pred)
    report = classification_report(all_truth, all_pred, target_names=label_list, labels=range(num_class), output_dict=True)
    report1 = classification_report(all_truth, all_pred, target_names=label_list, labels=range(num_class))
    cm = confusion_matrix(all_truth, all_pred, labels=range(num_class))
    
    save_base = os.path.splitext(eval_file)[0]
    if not os.path.exists(save_base):
        os.makedirs(save_base)
    with open(f"{save_base}/report.txt", "w") as f:
        f.write(json.dumps(report, indent=2))
    with open(f"{save_base}/report1.txt", "w") as f:
        f.write(report1)
    plot_confusion_matrix(cm, label_list, f"{save_base}/cm.png")
    print(f"{os.path.basename(eval_file).split('.')[0]}: {acc*100:.4f}%")

    # Save out-of-list embeddings for review
    if len(out_of_list_embeddings) > 0:
        with open(f"{save_base}/out_of_list.json", "w") as f:
            json.dump(out_of_list_embeddings, f, indent=2)
    if len(error_list) > 0:
        with open(f"{save_base}/error_list.json", "w") as f:
            json.dump(error_list, f, indent=2)
    
    return acc, num_sample

def infer(config_path, model):
    ['downstairs', 'jog', 'lie', 'sit', 'stand', 'upstairs', 'walk']
    mapping = {
        0: 'downstairs',
        1: 'jog',
        2: 'lie',
        3: 'sit',
        4: 'stand',
        5: 'upstairs',
        6: 'walk'
    }
    acc_total = {}
    num_total = {}
    config = yaml.safe_load(open(config_path))['TEST']
    test_paths = config['META']
    locs = config.get('LOC', default_loc)
    for i, loc in enumerate(locs):
        print(f"\t{i}. {loc}")
        config = {
            'META': test_paths,
            'LOC': [loc]
        }
        dataset = IMUDataset(config, is_train=False, is_rotated=False)
        if len(dataset) == 0:
            continue
        predictions = []
        correct_pred = 0
        
        # Create a DataLoader for batch processing
        batch_size = 512  # You can adjust this based on your memory constraints
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing ..."):
                labels, imu_inputs, location_embs, data_ids = batch
                imu_inputs = imu_inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if enable_aware:
                    location_embs = location_embs.to(device, non_blocking=True)
                else:
                    location_embs = None

                # Process the entire batch at once
                outputs = model(imu_inputs, prior_emb=location_embs)
                _, pred_indices = torch.max(outputs, 1)
                
                # Update correct predictions count
                correct_pred += (pred_indices == labels).sum().item()
                
                # Store predictions
                for pred_idx, label, data_id in zip(pred_indices, labels, data_ids):
                    predictions.append({
                        'pred': mapping[pred_idx.item()],
                        'ref': mapping[label.item()],
                        'data_id': data_id.item()
                    })

        result_file = load_path.split('/')[-2] + '_' + loc[:2] + '.json'
        prediction_file = os.path.join(save_path, result_file)
        json.dump(predictions, open(prediction_file, 'w'), indent=2)

        temp = eval(prediction_file)
        acc_total[result_file] = temp[0]
        num_total[result_file] = temp[1]
    print(json.dumps(acc_total, indent=2, sort_keys=True))
    print(json.dumps(num_total, indent=2, sort_keys=True))
    # weight acc
    total = sum([acc_total[k] * num_total[k] for k in acc_total.keys()])
    total_num = sum(num_total.values())
    print(f"Total Accuracy: {total / total_num * 100:.4f}%")
    # dump the results
    res = {
        'acc_total': acc_total,
        'num_total': num_total,
        'avg_acc': total / total_num
    }
    with open(os.path.join(save_path, 'eval_res.txt'), 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    # define the model
    if load_path.endswith('.pth'):
        load_file = os.path.basename(load_path)
        load_path = os.path.dirname(load_path)
    else:
        load_file = None

    model_args = json.load(open(os.path.join(load_path, 'args.json')))['model_args']
    model_args = UniTSArgs.from_dict(model_args)
    model = UniTS(enc_in=6, num_class=num_class, args=model_args)
    if load_file is None:
        record = json.load(open(os.path.join(load_path, 'best.json')))
        if enable_cross:
            epoch = record['lowest_epoch']
        else:
            epoch = record['best_epoch']
        load_file = f'checkpoint-{epoch}.pth'
    load_path = os.path.join(load_path, load_file)
    assert load_path is not None and os.path.exists(load_path)
    
    plot(os.path.dirname(load_path))

    pretrained_mdl = torch.load(load_path, map_location='cpu')
    msg = model.load_state_dict(pretrained_mdl['model'], strict=False)
    print(msg)
    
    model.to(device) # device is cuda
    # set trainable parameters

    for c_path in config_paths:
        infer(c_path, model)
        res = {
            'load_path': load_path,
            'config_path': c_path,
        }
        with open(os.path.join(save_path, 'config.txt'), 'w') as f:
            json.dump(res, f, indent=2)
    print(load_path)