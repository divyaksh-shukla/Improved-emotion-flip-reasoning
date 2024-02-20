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
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

from utils import *
from layers import *

class EFR_TX(nn.Module):
    def __init__(self):
        super(EFR_TX, self).__init__()
        EFR_CONFIG = CONFIG['efr_model']
        self.bert_tokenizer = BertTokenizer.from_pretrained(CONFIG['efr_embedding_model'], device=CONFIG['device'])
        self.bert = BertModel.from_pretrained(CONFIG['efr_embedding_model']).to(device=CONFIG['device'])
        self.tx_encoder = nn.TransformerEncoderLayer(d_model=EFR_CONFIG['encoder_dimension'], nhead=EFR_CONFIG['encoder_nhead'], device=CONFIG['device'])
        self.fc = nn.Linear(EFR_CONFIG['fc_in'], EFR_CONFIG['fc_out'], device=CONFIG['device'])
    
    def forward(self, context_utterance, context_emotion, target_utterance, target_emotion):
        x = context_utterance + ' [SEP] ' + context_emotion
        x = self.bert_tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=128).to(CONFIG['device'])
        context_embedding = self.bert(**x).pooler_output
        
        x = target_utterance
        x = self.bert_tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=128).to(CONFIG['device'])
        target_embedding = self.bert(**x).pooler_output
        
        x = torch.cat((context_embedding, target_embedding), dim=1)
        if (torch.isnan(x).any()):
            raise ContainsNansError
        x = self.tx_encoder(x) # The output of the transformer encoder
        x = self.fc(x)
        return x
    
class ERC_MMN(nn.Module):
    def __init__(self, erc_mmn_config_key=None):
        super(ERC_MMN, self).__init__()
        assert erc_mmn_config_key is not None, "ERC_MMN requires a configuration key"
        self.config = CONFIG[erc_mmn_config_key]
        
        self.interact_layer = interact(config_key='interact')
        self.memory_layer = MemoryNetwork(config_key='memory_network')
        self.pool = pool(config_key='pool')
        
        self.conversation_GRU = nn.GRU(input_size=self.config['conversation_GRU_in'], 
                                       hidden_size=self.config['conversation_GRU_out'], 
                                       num_layers=self.config['conversation_GRU_layers'], 
                                       batch_first=True, bidirectional=True)
        
        self.extra_GRU = nn.GRU(input_size=self.config['extra_GRU_in'], 
                               hidden_size=self.config['extra_GRU_out'], 
                               num_layers=self.config['extra_GRU_layers'], 
                               batch_first=True, bidirectional=True)
        
        self.linear = nn.Linear(self.config['linear_in'], self.config['linear_out'], device=CONFIG['device'])
        
    def forward(self, utterances, speakers):
        global_output, speaker_output = self.interact_layer(utterances, speakers)
        
        # memory_output = self.memory_layer(global_output, speaker_output)
        # memory_output = self.pool(memory_output)
        
        # memory_output = torch.cat([speaker_output, memory_output], dim=1)
        
        # conversation_GRU_output, _ = self.conversation_GRU(memory_output)
        
        # extra_GRU_output, _ = self.extra_GRU(conversation_GRU_output)
        # extra_GRU_output = extra_GRU_output.add(conversation_GRU_output)
        
        output = self.linear(global_output)
        
        return global_output, output
        