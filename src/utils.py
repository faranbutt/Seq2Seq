import torch
class Config:
    #model hyperparams
    EMB_DIM = 256
    N_LAYERS = 2
    HIDDEN_DIM = 512
    DROPOUT = 0.5


    #training hyperparams
    N_EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    CLIP = 1.0
    TEACHER_FORCING_RATIO = 0.5

    #tokenization
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'


    SRC_LANG = 'de'
    TGT_LANG = 'en'
    MIN_FREQ = 2

    MAX_LEN=50
    NUM_EVAL_SAMPLES=100

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG_DIR = 'runs/seq2seq_experiment'
    CHECKPOINT_DIR = 'checkpoints'
    SAVE_EVERY = 5


