import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import math

from utils import *

class UtteranceEmbedding(nn.Module):
    def __init__(self, utterence_embedding_config_key=None):
        super(UtteranceEmbedding, self).__init__()
        if utterence_embedding_config_key is None:
            utterence_embedding_config_key = 'utterance_embedding'
        self.config = CONFIG[utterence_embedding_config_key]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = AutoModel.from_pretrained(self.config['model_name'])
        self.embedding_dim = self.model.config.hidden_size
    
    def forward(self, utterance):
        tokens = self.tokenizer(utterance, return_tensors='pt', padding=True, truncation=True, max_length=128).to(CONFIG['device'])
        return self.model(**tokens).pooler_output
    
class DialogueGRU(nn.Module):
    def __init__(self, embedding_dim):
        super(DialogueGRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.config_key = 'dialogue_gru'
        self.config = CONFIG[self.config_key]
        self.embedding = UtteranceEmbedding()
        self.dGRU = nn.GRU(input_size=self.embedding_dim, hidden_size=self.config['hidden_size'], 
                           num_layers=self.config['num_layers'], batch_first=True, 
                           dropout=self.config['dropout'], bidirectional=self.config['bidirectional'])
        self.dropout = nn.Dropout(self.config['dropout'])
        
    def forward(self, utterances):
        num_utterances = len(utterances)
        hidden_size = self.config['hidden_size'] * 2 if self.config['bidirectional'] else self.config['hidden_size']
        outputs = torch.zeros((num_utterances, hidden_size)).to(CONFIG['device'])
        for i, utterance in enumerate(utterances):
            output = self.embedding(utterance)
            output, _ = self.dGRU(output)
            output = self.dropout(output)
            outputs[i, :] = output
        return outputs
    
class interact(nn.Module):
    def __init__(self, config_key=None):
        super(interact, self).__init__()
        assert config_key is not None, "interact requires a configuration key"
        self.config = CONFIG[config_key]
        
        self.hidden_size = self.config['hidden_dim']
        
        embedding_dim = self.config['embedding_dim']
        self.dialog_GRU = DialogueGRU(embedding_dim=768)   # Dialogue level
        
        self.global_GRU = nn.GRU(embedding_dim*3, self.hidden_size,1, 
                           batch_first=True, bidirectional=True)   #Global level
        
        self.attention = attention(query_embedding_dim=embedding_dim, key_embedding_dim=embedding_dim*2, hidden_dim=embedding_dim)
        
        self.speaker_GRU = nn.GRU(embedding_dim*2, embedding_dim, 1, batch_first=True, bidirectional=True)   #Speaker representation

    def forward(self, utterances, speakers):
        whole_dialogue_indices = utterances
               
        dialogue = self.dialog_GRU(whole_dialogue_indices)    # Get dialogue level representation
        batch_size = CONFIG['batch_size']
        utterance_len, embedding_dim = dialogue.size() # Get the dimensions of the dialogue
        
        device = dialogue.device
        
        # previous_global_GRU_output = torch.zeros((batch_size, utterance_len, embedding_dim)).to(device)
        global_output = torch.zeros((utterance_len, self.hidden_size * 2)).to(device)
        global_hidden = torch.randn(2, 1, self.hidden_size).to(device)
                    
        speaker_output = torch.zeros((utterance_len, embedding_dim * 2)).to(device)
        speaker_initial_hidden = torch.randn(2, 1, self.hidden_size).to(device)
        fop2 = torch.zeros((utterance_len, embedding_dim * 3)).to(device)
        initial_attention_output = torch.randn(1, 1, self.hidden_size).to(device) # Initial attention output, in the absence of any previous utterances
        
        # for batch_id in range(batch_size):
        # dialog_id = chat_ids[batch_id]
        speaker_hidden_states = {}
        for utterance_id in range(utterance_len):
            # previous_global_GRU_output = global_output.clone()
            
            current_utterance = dialogue[utterance_id, :]
            
            current_speaker = speakers[utterance_id]
            
            if current_speaker not in speaker_hidden_states:
                speaker_hidden_states[current_speaker] = speaker_initial_hidden
            
            speaker_hidden = speaker_hidden_states[current_speaker]
            current_utterance_embedding = torch.unsqueeze(torch.unsqueeze(current_utterance, 0), 0)
            
            # Attention Key:
            # Taking all the previous utterances (including the current one), only required in the else condition below
            # The keys are comming from the output of the global_GRU
            # key = previous_global_GRU_output[batch_id][:utterance_id+1].clone() 
            key = global_output[:utterance_id+1, :]
            key = torch.unsqueeze(key,0)
            
            if utterance_id == 0: # If it is the first utterance in the dialogue
                tmp = torch.cat([initial_attention_output, current_utterance_embedding], -1).to(device)
                speaker_output[utterance_id, :], updated_speaker_hidden_state = self.speaker_GRU(tmp,speaker_hidden)
            else:
                query = current_utterance_embedding
                attention_output,_ = self.attention(key,query) # Attention between current and previous utterances
                
                # Concatenating the attention output and the current utterance for input to speaker_GRU
                tmp = torch.cat([attention_output, current_utterance_embedding], -1).to(device) 
                speaker_output[utterance_id, :], updated_speaker_hidden_state = self.speaker_GRU(tmp,speaker_hidden)
            
            speaker_output[utterance_id, :] = speaker_output[utterance_id, :].add(tmp)        # Residual Connection        
            speaker_hidden_states[current_speaker] = updated_speaker_hidden_state
            
            fop2[utterance_id, :] = torch.cat([speaker_output[utterance_id, :],dialogue[utterance_id, :]], -1)
            tmp = torch.unsqueeze(torch.unsqueeze(fop2[utterance_id, :], 0), 0)
            global_output[utterance_id, :], global_hidden = self.global_GRU(tmp, global_hidden)
            
        # del speaker_hidden_states, speaker_hidden, current_utterance, current_speaker, current_utterance_embedding, key, query, attention_output, tmp, updated_speaker_hidden_state, fop2, initial_attention_output, speaker_initial_hidden
        return global_output,speaker_output

class MaskedAttention(nn.Module):
    def __init__(self, query_embedding_dim, key_embedding_dim=None, value_embeddim_dim=None, hidden_dim=None, dropout=None):
        super(MaskedAttention, self).__init__()
        self.query_embedding_dim = query_embedding_dim
        self.key_embedding_dim = key_embedding_dim if key_embedding_dim is not None else query_embedding_dim
        self.value_embedding_dim = value_embeddim_dim if value_embeddim_dim is not None else key_embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else key_embedding_dim
        dropout = dropout if dropout is not None else 0.1
        
        self.attn = nn.MultiheadAttention(embed_dim=self.query_embedding_dim, kdim=self.key_embedding_dim, vdim=self.value_embedding_dim, num_heads=1, dropout=dropout)
    
    def forward(self, key, query, mask=None):
        if mask is None:
            mask = torch.zeros((query.size()[0], query.size()[0])).to(query.device)
            for i in range(mask.size()[0]):
                mask[i, :i+1] = 1
            # mask = mask.repeat(query.size()[0], 1, 1)
        output, score = self.attn(query, key, key, attn_mask=mask)
        return output, score

class attention(nn.Module):
    def __init__(self, query_embedding_dim, key_embedding_dim=None, value_embeddim_dim=None, hidden_dim=None, dropout=None):
        super(attention, self).__init__()
        self.query_embedding_dim = query_embedding_dim
        self.key_embedding_dim = key_embedding_dim if key_embedding_dim is not None else query_embedding_dim
        self.value_embedding_dim = value_embeddim_dim if value_embeddim_dim is not None else query_embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.key_embedding_dim
        
        self.w_q = nn.Linear(self.query_embedding_dim, self.hidden_dim)
        self.w_k = nn.Linear(self.key_embedding_dim, self.hidden_dim)
        self.w_v = nn.Linear(self.value_embedding_dim, self.hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Dropout(0.0)
        self.normalizing_factor = math.sqrt(self.hidden_dim)
    
    def _mask_score(self, score, mask):
        score = score.masked_fill(mask == 0, -1e9)
        return score
    
    def forward(self, key, query, mask=None):
        if (query.dim() == 1):
            query = torch.unsqueeze(query, 0)
        if (key.dim() == 1):
            key = torch.unsqueeze(key, 0)
        
        if (query.dim() == 2):
            query = torch.unsqueeze(query, 1)
        if (key.dim() == 2):
            key = torch.unsqueeze(key, 1)
        
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(key)
        
        score = torch.bmm(query, key.transpose(1, 2))/self.normalizing_factor
        
        if mask is not None:
            score = self._mask_score(score, mask)
        
        score = F.softmax(score, dim=2)
        # score.data[score!=score] = 0         #removing nan values
        
        output = torch.bmm(score, value)
        return output, score


class pool(nn.Module):
    def __init__(self, mode="mean", config_key=None):
        super(pool,self).__init__()
        assert config_key is not None, "pool requires a configuration key"
        self.config = CONFIG[config_key]
        self.mode = mode
    def forward(self, x):
        device = x.device
        op = torch.zeros((x.size()[0],x.size()[1])).to(device)
        this_tensor = []
        for s in range(x.size()[0]):
            this_tensor.append(x[s, :])
            if self.mode == "mean":
                op[s, :] = torch.mean(torch.stack(this_tensor),0)
            elif self.mode == "max":
                op[s, :],_ = torch.max(torch.stack(this_tensor),0)
            elif self.mode == "sum":
                op[s, :] = torch.sum(torch.stack(this_tensor),0)
            else:
                print("Error: Mode can be either mean or max only")
        return op
    
class MemoryNetwork(nn.Module):
  def __init__(self, config_key=None):
    super(MemoryNetwork,self).__init__()
    self.config = CONFIG[config_key]
    num_hops = self.config['num_hops']
    hidden_size = self.config['hidden_size']
    batch_size = CONFIG['batch_size']
    seq_len = CONFIG['seq_len']
    query_dim = self.config['query_dim']
    key_dim = self.config['key_dim']
    value_dim = self.config['value_dim']
    self.num_hops = num_hops
    self.memory_GRU = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
    self.masked_attention = MaskedAttention(query_dim, key_dim, value_dim, hidden_size).to(CONFIG['device'])
  
  def forward(self, global_output, speaker_output):
    memory_output = global_output
    for hop in range(self.num_hops):
      dialogue,h = self.memory_GRU(memory_output)
      memory_output, _ = self.masked_attention(dialogue, speaker_output)
    return memory_output