from tabnanny import check
import torch
import torch.nn as nn
from src.utils import Config
from src.data_utils import DataProcessor
from src.model import Encoder, Decoder, Seq2Seq
from src.train import Trainer
from src.evaluate import Evaluator
import os

def main():
    config = Config()
    trainer=  Trainer()
    print(f"Using device: {config.DEVICE}")
    print("\n" + "="*80)
    print("Loading and Processing Data")
    print('='*80)
    data_processor = DataProcessor(config)
    train_loader, val_loader, train_data, val_data =  data_processor.get_dataloaders()
    print('\n'+'='*80)
    print("Building Model")
    print('='*80)
    
    encoder = Encoder(
        n_tokens = len(data_processor.src_vocab),
        emb_dim=config.EMB_DIM,
        hid_dim=config.HIDDEN_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT,
        pad_idx=data_processor.src_vocab[config.PAD_TOKEN]
    )
    decoder = Decoder(
         n_tokens = len(data_processor.src_vocab),
        emb_dim=config.EMB_DIM,
        hid_dim=config.HIDDEN_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT,
        pad_idx=data_processor.trg_vocab[config.PAD_TOKEN]
    )
    
    model = Seq2Seq(encoder,decoder).to(config.DEVICE)
    model.init_weights()
    
    print(f"Model has {model.count_parameters():,} trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(
        ignore_index = data_processor.trg_vocab[config.PAD_TOKEN]
    )   
    print("\n" + "="*80)
    print("Training Model")
    print("="*80)
    
    trainer.train(train_loader,val_loader)
    print("\n" + "="*80)
    print("Evaluating Model")
    print("="*80)
    
    evaluator = Evaluator(model,data_processor,config, config.DEVICE)
    evaluator.evaluate_samples(val_data,num_samples=10)
    
    bleu_score = evaluator.calculate_bleu(val_data, num_samples=config.NUM_EVAL_SAMPLES)
    print(f"\n{'='*80}")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"{'='*80}\n")
    
def translate_interactive():
    config=  Config()
    print("Loading vocablaries....")
    data_processor = DataProcessor(config)
    train_loader, val_loader, train_data, val_data = data_processor.get_data_loaders()
    
    encoder = Encoder(
        n_tokens=len(data_processor.src_vocab),
        emb_dim=config.EMB_DIM,
        hid_dim=config.HID_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT,
        pad_idx=data_processor.src_vocab[config.PAD_TOKEN]
    )
    
    decoder = Decoder(
        n_tokens=len(data_processor.trg_vocab),
        emb_dim=config.EMB_DIM,
        hid_dim=config.HID_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT,
        pad_idx=data_processor.trg_vocab[config.PAD_TOKEN]
    )
    
    model = Seq2Seq(encoder, decoder).to(config.DEVICE)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR,'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        print("Please train the model first by running: python main.py")
        return
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path,map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    evaluator = Evaluator(model,data_processor,config, config.DEVICE)
   
   
    print("\n" + "="*80)
    print("Interactive Translation Mode (German to English)")
    print("Enter 'quit' to exit")
    print("="*80 + "\n")
    
    while True:
        src_sentence = input("German: ").strip()
        
        if src_sentence.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not src_sentence:
            continue
        
        translation = evaluator.translate(src_sentence)
        print(f"English: {translation}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'translate':
        translate_interactive()
    else:
        main()