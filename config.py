import torch

CONFIG = {
    'model_type': 'erc',        # The type of model to train. Can be 'efr' or 'erc
    'data_dir': 'data',         # The directory where the data is stored
    
    'efr_train_file': 'MELD_train_efr.json',   # The name of the data file
    # 'efr_train_file': 'MaSaC_train_efr.json',   # The name of the data file
    
    'efr_test_file': 'MELD_test_efr.json',   # The name of the data file
    # 'efr_test_file': 'MaSaC_test_efr.json',   # The name of the data file
    
    'efr_val_file': 'MELD_val_efr.json',   # The name of the data file
    # 'efr_val_file': 'MaSaC_val_efr.json',   # The name of the data file
    
    'erc_train_file': 'MaSaC_train_erc.json',   # The name of the data file
    
    'emotion_map': {
        'neutral': 0,
        'joy': 1,
        'sadness': 2,
        'fear': 3,
        'anger': 4,
        'disgust': 5,
        'surprise': 6
    },
    
    'efr_embedding_model': 'l3cube-pune/hing-bert',  # The name of the BERT model to use for EFR embedding
    'batch_size': 8,  # The batch size for training
    'seq_len': 20,  # The sequence length for the input data
    
    'efr_model': {
        'encoder_dimension': 768*2,  # The dimension of the encoder
        'encoder_nhead': 8,                   # The number of heads in the multihead attention
        'fc_in': 768*2,               # The input dimension of the fully connected layer
        'fc_out': 1,                  # The output dimension of the fully connected layer
    },
    
    'erc_mmn': {
    },
    
    'interact': {
        'hidden_dim': 768,  # The hidden dimension of the interact layer
        'embedding_dim': 768,  # The dimension of the embeddings        
    },
    
    'memory_network': {
        'num_hops': 3,  # The number of hops in the memory network
        'hidden_size': 768,  # The hidden size of the memory network
    },
    
    'pool': {
    },
    
    'dialogue_gru': {
        'hidden_size': 768,  # The hidden size of the GRU
        'num_layers': 1,  # The number of layers in the GRU
        'bidirectional': True,  # Whether the GRU is bidirectional
        'dropout': 0.1,  # The dropout probability
    },
    
    'tensorboard_dir': 'runs',  # The directory where the tensorboard logs are stored
    'model_dir': 'models',      # The directory where the trained models are stored
    'model_name': 'EFR_TX',     # The name of the model to be saved
    'model_ext': 'pt',          # The extension of the model file
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),  # The device to use for training
}