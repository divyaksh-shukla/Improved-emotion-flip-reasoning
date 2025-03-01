import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel
from model import EFR_TX, ERC_MMN
from tqdm import tqdm
import yaml
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from utils import *

if not os.path.exists(CONFIG['tensorboard_dir']):
    os.makedirs(CONFIG['tensorboard_dir'])
    if not os.path.exists(os.path.join(CONFIG['tensorboard_dir'], CONFIG['model_name'])):
        os.makedirs(os.path.join(CONFIG['tensorboard_dir'], CONFIG['model_name']))

if not os.path.exists(CONFIG['model_dir']):
    os.makedirs(CONFIG['model_dir'])

tb_logger = tensorboardX.SummaryWriter(os.path.join(CONFIG['tensorboard_dir'], CONFIG['model_name']))
tb_logger.add_hparams(CONFIG, {}, CONFIG['model_name'])

# Load the data
def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data



def train_efr_tx(model, train_data, epochs=1, val_data=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    main_pbar = tqdm(total=epochs*len(train_data), bar_format="{l_bar}{bar:20}{r_bar}")
    for epoch in range(epochs):
        for dialog_id, sub_dialog in enumerate(train_data):
            context_utterances = sub_dialog['utterances']
            context_emotion = sub_dialog['emotions']
            target_utterance = sub_dialog['utterances'][-1]
            target_emotion = sub_dialog['emotions'][-1]
            trigger_targets = sub_dialog['triggers']
            
            optimizer.zero_grad()
            for i, (context_utterance, context_emotion, trigger_target) in enumerate(zip(context_utterances, context_emotion, trigger_targets)):
                try:
                    trigger_pred = model(context_utterance, context_emotion, target_utterance, target_emotion)
                except ContainsNansError:
                    save_model_and_exit(model, model_name=CONFIG['model_name']+'_1')
                loss = criterion(trigger_pred.squeeze(1), torch.tensor([trigger_target], device=CONFIG['device']))
                loss /= len(trigger_targets)
                loss.backward()
                
            optimizer.step()
            main_pbar.set_description(f'Training Epoch {epoch} | Dialog {dialog_id} ')
            main_pbar.set_postfix({
                'Loss': f"{loss.item():.5f}"
            })
            tb_logger.add_scalar('Train Loss', loss.item(), epoch*len(train_data) + dialog_id)

def train_erc_mmn(model, train_data, epochs=1, val_data=None):
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            main_pbar = tqdm(total=len(train_data), bar_format="{l_bar}{bar:20}{r_bar}")
            for dialog_id, sub_dialog in enumerate(train_data):
                # if len(sub_dialog['utterances']) > 5:
                #     continue
                speakers = sub_dialog['speakers']
                utterances = sub_dialog['utterances']
                emotions = sub_dialog['emotions']
                emotions = [CONFIG['emotion_map'][emotion] for emotion in emotions]
                target_emotions = torch.tensor(emotions, device=CONFIG['device'])
                
                optimizer.zero_grad()
                _, predicted_emotions = model(utterances, speakers)
                loss = criterion(predicted_emotions, target_emotions)
                loss.backward()
                    
                optimizer.step()
                main_pbar.set_description(f'Training Epoch {epoch} | Dialog {dialog_id} ')
                main_pbar.set_postfix({
                    'Loss': f"{loss.item():.5f}",
                    'utterance length': f"{len(utterances)}",
                    "Memory Allocated": f"{torch.cuda.memory_allocated(CONFIG['device']) / 1e9:.2f}GB"
                })
                main_pbar.update(1)
                tb_logger.add_scalar('Train Loss', loss.item(), epoch*len(train_data) + dialog_id)
            model.save(os.path.join(CONFIG['model_dir'], CONFIG['model_name']+'_1.pth'))

def evaluate_erc_mmn(model, validation_data, model_filepath=None):
    model.eval()
    if model_filepath:
        model.load(model_filepath)
    with torch.inference_mode():
        main_pbar = tqdm(total=len(validation_data), bar_format="{l_bar}{bar:20}{r_bar}")
        for dialog_id, sub_dialog in enumerate(validation_data):
            speakers = sub_dialog['speakers']
            utterances = sub_dialog['utterances']
            emotions = sub_dialog['emotions']
            emotions = [CONFIG['emotion_map'][emotion] for emotion in emotions]
            # target_emotions = torch.tensor(emotions, device=CONFIG['device'])
            
            _, predicted_emotions = model(utterances, speakers)
            predicted_emotions = F.softmax(predicted_emotions, dim=1).argmax(1).cpu().numpy()
            
            accuracy = accuracy_score(emotions, predicted_emotions)
            F1 = f1_score(emotions, predicted_emotions, average='weighted')
            
            main_pbar.set_description(f'Evaluation | Dialog {dialog_id} ')
            main_pbar.set_postfix({
                'Accuracy': f"{accuracy:.5f}",
                'F1': f"{F1:.5f}"
            })
            main_pbar.update(1)
            tb_logger.add_scalar('Accuracy', accuracy, dialog_id)
            tb_logger.add_scalar('F1', F1, dialog_id)
            
def predict_erc_mmn(model, test_data, answer_filepath):
    model.eval()
    with torch.inference_mode():
        with open(answer_filepath, 'w') as f:
            reverse_emotion_map = {v: k for k, v in CONFIG['emotion_map'].items()}
            main_pbar = tqdm(total=len(test_data), bar_format="{l_bar}{bar:20}{r_bar}")
            for dialog_id, sub_dialog in enumerate(test_data):
                speakers = sub_dialog['speakers']
                utterances = sub_dialog['utterances']
                # emotions = sub_dialog['emotions']
                # emotions = [CONFIG['emotion_map'][emotion] for emotion in emotions]
                # target_emotions = torch.tensor(emotions, device=CONFIG['device'])
                
                _, predicted_emotions = model(utterances, speakers)
                predicted_emotions = torch.argmax(F.softmax(predicted_emotions, dim=0), dim=0).cpu().numpy()
                predicted_emotions = [reverse_emotion_map[emotion] for emotion in predicted_emotions]
                f.writelines(predicted_emotions)
                f.flush()
                
                main_pbar.set_description(f'Evaluation | Dialog {dialog_id} ')
                main_pbar.update(1)

if __name__ == '__main__':
    if CONFIG['model_name'] == 'EFR_TX':
        train_data = load_data(os.path.join(CONFIG['data_dir'], CONFIG['efr_train_file']))
        model = EFR_TX().to(CONFIG['device']) # Create the model
        train_efr_tx(model, train_data, epochs=1)
    elif CONFIG['model_name'] == 'ERC_MMN':
        train_data = load_data(os.path.join(CONFIG['data_dir'], CONFIG['erc_train_file']))
        val_data = load_data(os.path.join(CONFIG['data_dir'], CONFIG['erc_val_file']))
        model = ERC_MMN('erc_mmn').to(CONFIG['device'])
        # train_erc_mmn(model, train_data, epochs=1)
        evaluate_erc_mmn(model, val_data, model_filepath=os.path.join(CONFIG['model_dir'], CONFIG['model_name']+'_1.pth'))