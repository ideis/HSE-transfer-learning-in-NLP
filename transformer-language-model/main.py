import random
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam

# % pip install sentencepiece
import sentencepiece as spm

from transformer_language_model import TransformerCharLM
from utils import generate_data, clean_text, tokenize_text

def train():
    model.train()
    losses = list()
    for batch_id, batch in enumerate(generate_data(text_as_int, batch_size, seq_len)):
        src, trg = batch
        src = src.permute(1,0).to(device)
        trg = trg.permute(1,0).to(device)
        optimizer.zero_grad()
        preds = model(src)
        loss = criterion(preds.contiguous().view(-1, vocab_size), trg.contiguous().view(-1))
        losses.append(loss.item())
        avg_loss = sum(losses)/len(losses)
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
          print(f'epoch: {epoch} | loss: {avg_loss:.4f}')

def generate(n_chars=500, temperature=0.1):
    model.eval()
    while True:
      src = sp.EncodeAsIds(random.choice(string.ascii_uppercase))
      src = torch.tensor(src).unsqueeze(0).to(device)
      if len(src) == 1: # hack
        break
    for i in range(n_chars):
        preds = model(src)
        logits = preds.squeeze(1)
        logits = logits[-1, :].div(temperature)
        probs = F.softmax(logits, dim=-1)
        char_idx = torch.multinomial(probs, num_samples=1)
        src = torch.cat([src, char_idx.unsqueeze(-1)], dim=0).to(device)
    text = sp.DecodeIds(src.squeeze().tolist())
    print(text)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'device: {device}')  

seq_len = 256
vocab_size = 4000
batch_size = 64
lr = 0.001
epochs = 500

n_chars=1000 # to generate
temperature=0.7

text_file = 'fiction.txt'

spm.SentencePieceTrainer.Train(f'--input={text_file} --model_prefix=tokens --vocab_size={vocab_size}')
sp = spm.SentencePieceProcessor()
sp.Load("tokens.model")

text = open(text_file, 'rb').read().decode(encoding='utf-8')
text = clean_text(text)
text_as_int = np.array(sp.EncodeAsIds(text))

model = TransformerCharLM(vocab=vocab_size, d_model=384, n_heads=8, n_encoder_layers=10, d_ff=2048, dropout=0.1)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

try:
    for epoch in range(1, epochs+1):
        train()
        print('-' * 89)
        generate(n_chars, temperature)
        print('-' * 89)

        # Save the model at each 50 epoch.
        save_path = 'model.pt'
        if epoch == 50:
          with open(save_path + str(epoch), 'wb') as f:
              torch.save(model, f)

except KeyboardInterrupt:
    print('-' * 89)
    print('exiting from training early')