import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
import os


class Trainer:
    def __init__(self,model,optimizer,criterion,config, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config 
        self.device = device
        self.writer = SummaryWriter(config.LOG_DIR)
        self.global_step = 0
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok = True)
        
        
    def train_epoch(self,train_loader):
        self.model.train()
        epoch_loss = 0
        for src,trg in tqdm(train_loader,desc="Training",leave=False):
            src,trg = src.to(self.device), trg.to(self.device)
            output = self.model(src,trg,self.config.TEACHER_FORCING_RATIO)
            output = output.view(-1,output.shape[-1])
            trg = trg[1:].view(-1)
            loss = self.criterion(output, trg)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(self.model.parameters(),self.config.CLIP)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.writer.add_scalar("Training/batch_loss",loss.item(),self.global_step)
            self.global_step += 1
        return epoch_loss / len(train_loader)
    
    def evaluate(self,val_loader):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for src,trg in tqdm(val_loader, desc = "Validation", leave = False):
                src, trg = src.to(self.device),   trg.to(self.device)
                output = self.model(src,trg, teacher_forcing_ratio = 0)
                output = output.view(-1,output.shape[-1])
                trg = trg[1:].view(-1)
                loss = self.criterion(output,trg)
                epoch_loss += loss.item()
        return epoch_loss / len(val_loader)
    def train(self, train_loader, val_loader):
        best_val_loss = float('inf') 
        for epoch in trange(self.config.N_EPOCHS, desc="Epochs"):
            train_loss = self.train_epoch(train_loader)
            self.writer.add_scalar("Loss/train",train_loss, epoch)
            val_loss = self.evaluate(val_loader)
            self.write.add_scalar('Loss/validation',val_loss,epoch)
            
            print(f'n\Epoch {epoch+1}/{self.config.N_EPOCHS}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}') 
            
            
            if (epoch + 1) % self.config.SAVE_ENTRY == 0:
                self.save_checkpoint(epoch,train_loss,val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                self.save_checkpoint(epoch, train_loss, val_loss, is_best = True)
        self.writer.close()
        print(f'\n Training complete ! Best validation loss: {best_val_loss:.4f}')
    def save_checkpoint(self,epoch,train_loss, val_loss, is_best=False):
        checkpoint  = {
            'epoch' : epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        if is_best:
            path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pt')
            print(f"Saving best model to {path}")
        else:
            path = os.path.join(self.config.CHECKPOINT_DIR, f'check_epoch_{epoch+1}.pt')
        torch.save(checkpoint,path)
        
    def load_checkpoint(self,path):
        checkpoint = torch.load(path,map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']
    