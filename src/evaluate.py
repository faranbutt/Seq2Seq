import model
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class Evaluator:
    def __init__(self,model, data_processor, config, device) -> None:
        self.model = model
        self.data_processor  = data_processor
        self.config = config
        self.device = device
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    def translate(self, src_sentence):
        self.model.eval()
        encoded = self.data_processor.encode(
            src_sentence,
            self.data_processor.src_vocab,
            reverse = True
        )
        
        src_tensor = torch.tensor(encoded).unsqueeze(1).to(self.device)
        with torch.no_grad():
            hidden = self.model.encoder(src_tensor)
            
        trg_vocab = self.data_processor.trg_vocab
        trg_itos = self.vocab.get_itos()
        pred_tokens = [trg_vocab[self.config.SOS_TOKEN]]
        
        with torch.no_grad():
            for _ in range(self.config.MAX_LEN):
                decoder_input = torch.tensor([pred_tokens[-1]]).to(self.device)
                pred,hidden = self.model.decoder(decoder_input,hidden)
                _, pred_token = pred.max(dim=1)
                if pred_token == trg_vocab[self.config.EOS_TOKEN]:
                    break
                pred_tokens.append(pred_token)
                
        pred_words = [trg_itos[i] for i in pred_tokens[1:]]
        return ' '.join(pred_tokens)

        
    def evaluate_samples(self,data,num_samples=10):
        self.model.eval()
        print('\n'+'='*80)
        print("Sample Translations")
        print('='*80 + '\n')
        for i, (src, ref) in enumerate(data[:num_samples]):
            translation = self.translate(src)
            print(f"Example {i+1}:")
            print(f"Source:      {src.rstrip().lower()}")
            print(f"Reference:   {ref.rstrip().lower()}")
            print(f"Translation: {translation}")
            print()
            
    def calculate_bleu(self,data,num_samples=None):
        self.model.eval()
        if num_samples is None:
            num_samples = len(data)
        trg_vocab = self.data_processor.trg_vocab
        trg_itos = trg_vocab.get_itos()
        smoothie = SmoothingFunction().method4
        references = []
        hypotheses = []
        
        print(f"\nCalculating BLEU score on {num_samples} samples...")
        
        with torch.no_grad():
            for i, (src, ref) in enumerate(data[:num_samples]):
                # Encode source
                encoded = self.data_processor.encode(
                    src,
                    self.data_processor.src_vocab,
                    reverse=True
                )
                src_tensor = torch.tensor(encoded).unsqueeze(1).to(self.device)
                
                hidden = self.model.encoder(src_tensor)
                
                pred_tokens = [trg_vocab[self.config.SOS_TOKEN]]
                for _ in range(self.config.MAX_LEN):
                    decoder_input = torch.tensor([pred_tokens[-1]]).to(self.device)
                    pred, hidden = self.model.decoder(decoder_input, hidden)
                    _, pred_token = pred.max(dim=1)
                    pred_token = pred_token.item()
                    
                    if pred_token == trg_vocab[self.config.EOS_TOKEN]:
                        break
                    
                    pred_tokens.append(pred_token)
                

                pred_words = [trg_itos[j] for j in pred_tokens[1:]]
                ref_words = self.data_processor.tokenize(ref)
                
                references.append([ref_words])
                hypotheses.append(pred_words)
                
               
                if i < 5:
                    print(f"\nExample {i+1}:")
                    print(f"Source: {src}")
                    print(f"Reference: {ref}")
                    print(f"Translation: {' '.join(pred_words)}")
        
        bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
        return bleu * 100

        