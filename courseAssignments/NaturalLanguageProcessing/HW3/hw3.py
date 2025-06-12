!pip install transformers datasets spacy
!pip install -U "fsspec==2023.6.0"

!python -m spacy download en_core_web_sm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import spacy
from collections import Counter
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
VOCAB_SIZE = len(tokenizer.get_vocab())
nlp = spacy.load("en_core_web_sm")

# Load IMDb and split train/dev
imdb = load_dataset("imdb")
split = imdb["train"].train_test_split(0.2)
imdb["train"], imdb["dev"] = split["train"], split["test"]
del imdb["unsupervised"]

# Tokenize using tokenizer
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True, remove_columns="text")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# DataLoaders
train_dataloader = DataLoader(tokenized_imdb["train"], shuffle=True, collate_fn=data_collator, batch_size=64)
eval_dataloader = DataLoader(tokenized_imdb["dev"], collate_fn=data_collator, batch_size=64)
test_dataloader = DataLoader(tokenized_imdb["test"], collate_fn=data_collator, batch_size=64)

class Convolutional(nn.Module):
    def __init__(self, embedding_dim, kernel_sizes, filters, output_dim):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.6),
                nn.Conv1d(embedding_dim, filters, k),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        self.linear = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(len(kernel_sizes) * filters, output_dim)
        )

    def forward(self, batch):
        embedded = self.embeddings(batch["input_ids"]).transpose(1, 2)
        max_pooled = torch.cat([conv(embedded).max(dim=2).values for conv in self.conv], dim=1)
        return self.linear(max_pooled)


@torch.no_grad()
def binary_accuracy(preds, y):
    return (torch.round(torch.sigmoid(preds)) == y).float().mean()

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_acc, steps = 0, 0, 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(batch).squeeze(1)
        loss = criterion(preds, batch["labels"].float())
        acc = binary_accuracy(preds, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
        steps += 1
    return total_loss / steps, total_acc / steps

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_acc, steps = 0, 0, 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(batch).squeeze(1)
        loss = criterion(preds, batch["labels"].float())
        acc = binary_accuracy(preds, batch["labels"])
        total_loss += loss.item()
        total_acc += acc.item()
        steps += 1
    return total_loss / steps, total_acc / steps

model = Convolutional(embedding_dim=200, kernel_sizes=[1,2,3,4,5], filters=10, output_dim=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss().to(device)

for epoch in range(10):  # increase to 20 for higher accuracy
    train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, eval_dataloader, criterion)
    print(f"Epoch {epoch+1:02} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "model_cnn.pt")

embedding_matrix = model.embeddings.weight.data  # [vocab_size, emb_dim]
conv1_weights = model.conv[0][1].weight.data.squeeze(-1)  # [filters, emb_dim]

filter_responses = torch.matmul(embedding_matrix, conv1_weights.T)  # [vocab_size, num_filters]

top_k = 10
top_scores, top_indices = torch.topk(filter_responses, k=top_k, dim=0)

top_tokens_per_filter = []
for i in range(conv1_weights.size(0)):
    token_ids = top_indices[:, i].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    top_tokens_per_filter.append(tokens)
    print(f"Filter {i}: {tokens}")

all_top_tokens = sum(top_tokens_per_filter, [])
all_top_tokens_cleaned = [t.replace("##", "") for t in all_top_tokens]

pos_counter = Counter()
for tok in all_top_tokens_cleaned:
    doc = nlp(tok)
    if doc:
        pos_counter[doc[0].pos_] += 1

pos_df = pd.DataFrame(pos_counter.items(), columns=["POS", "Frequency"]).sort_values("Frequency", ascending=False)
pos_df.reset_index(drop=True, inplace=True)
pos_df

