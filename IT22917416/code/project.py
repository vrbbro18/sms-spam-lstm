"""One-model SMS spam classification. Run train, then predict. CPU by default."""
import argparse
import copy
import io
import json
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
DATA = ROOT / 'data' / 'SMSSpamCollection'
URL = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
SEED = 42
MAX_LEN = 100


def tokens(text):
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())


def encode(texts, vocab):
    rows = [([vocab.get(t, 1) for t in tokens(s)][:MAX_LEN] or [1]) for s in texts]
    lengths = torch.tensor([len(r) for r in rows])
    padded = torch.zeros((len(rows), MAX_LEN), dtype=torch.long)
    for i, row in enumerate(rows):
        padded[i, :len(row)] = torch.tensor(row)
    return padded, lengths


class SpamLSTM(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = nn.Embedding(size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(64, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(hidden[-1])).squeeze(1)


def load_data():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading UCI SMS Spam Collection...', flush=True)
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                DATA.write_bytes(archive.read('SMSSpamCollection'))
        except Exception as exc:
            raise SystemExit(f'Download failed: {exc}\nDownload the ZIP from the UCI dataset page and put SMSSpamCollection inside {DATA.parent}')
    rows = []
    for line in DATA.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            label, message = line.split('\t', 1)
            if label not in {'ham', 'spam'}:
                raise ValueError(f'Unexpected label: {label}')
            rows.append((label, message))
    df = pd.DataFrame(rows, columns=['label', 'text'])
    raw = len(df)
    # Identical model inputs must never occur in different splits.
    df['key'] = df.text.map(lambda s: ' '.join(tokens(s)))
    conflicting = df.groupby('key').label.nunique()
    bad = set(conflicting[conflicting > 1].index)
    df = df[~df.key.isin(bad) & df.key.ne('')].drop_duplicates('key').reset_index(drop=True)
    df['target'] = (df.label == 'spam').astype(int)
    print(f'Raw rows: {raw}; unique usable messages: {len(df)}')
    return df, raw


def loader(frame, vocab, shuffle=False):
    x, lengths = encode(frame.text.tolist(), vocab)
    y = torch.tensor(frame.target.to_numpy(), dtype=torch.float32)
    return DataLoader(TensorDataset(x, lengths, y), batch_size=64, shuffle=shuffle)


def evaluate(model, batches, loss_fn):
    model.eval()
    losses, ys, scores = [], [], []
    with torch.no_grad():
        for x, lengths, y in batches:
            logits = model(x, lengths)
            losses.append((loss_fn(logits, y).item(), len(y)))
            ys.extend(y.tolist())
            scores.extend(torch.sigmoid(logits).tolist())
    return sum(a*b for a,b in losses)/sum(b for _,b in losses), np.array(ys), np.array(scores)


def train(epochs):
    OUT.mkdir(exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    df, raw = load_data()
    train_df, rest = train_test_split(df, test_size=0.30, stratify=df.target, random_state=SEED)
    val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest.target, random_state=SEED)
    counts = Counter(t for s in train_df.text for t in tokens(s))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in counts.most_common(7998):
        if count >= 2:
            vocab[word] = len(vocab)
    batches = loader(train_df, vocab, True)
    validation = loader(val_df, vocab)
    model = SpamLSTM(len(vocab))
    # Training-only class weighting offsets the smaller spam class.
    positive_weight = float((train_df.target == 0).sum() / (train_df.target == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best, stale, best_epoch = [], float('inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for x, lengths, y in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
        vl, _, _ = evaluate(model, validation, loss_fn)
        tl = total / len(train_df)
        history.append({'epoch': epoch, 'training_loss': tl, 'validation_loss': vl})
        print(f'Epoch {epoch:02d}: train loss={tl:.4f}, validation loss={vl:.4f}', flush=True)
        if vl < best:
            best, stale, best_epoch = vl, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 3:
                print('Early stopping.')
                break
    duration = time.perf_counter() - started
    model.load_state_dict(best_state)
    torch.save({'state': best_state, 'vocab': vocab, 'max_len': MAX_LEN}, OUT / 'model.pt')
    pd.DataFrame(history).to_csv(OUT / 'history.csv', index=False)
    results = {}
    for name, frame in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        _, labels, probabilities = evaluate(model, loader(frame, vocab), loss_fn)
        predictions = (probabilities >= 0.5).astype(int)
        results[name] = classification_report(labels, predictions, target_names=['ham', 'spam'], output_dict=True, zero_division=0)
        results[name]['roc_auc'] = float(roc_auc_score(labels, probabilities))
        if name == 'test':
            print('\nFINAL TEST RESULTS\n' + classification_report(labels, predictions, target_names=['ham', 'spam'], zero_division=0))
            ConfusionMatrixDisplay(confusion_matrix(labels, predictions), display_labels=['ham','spam']).plot()
            plt.title('Held-out test set'); plt.tight_layout(); plt.savefig(OUT / 'confusion_matrix.png'); plt.close()
            pd.DataFrame({'message': frame.text.to_numpy(), 'actual': labels.astype(int), 'predicted': predictions, 'spam_score': probabilities}).to_csv(OUT / 'test_predictions.csv', index=False)
    config = {'seed': SEED, 'max_length': MAX_LEN, 'vocabulary_size': len(vocab), 'embedding_dim': 64, 'hidden_dim': 64, 'dropout': 0.3, 'batch_size': 64, 'learning_rate': 0.001, 'spam_positive_weight': positive_weight, 'threshold': 0.5, 'best_epoch': best_epoch, 'training_seconds': duration, 'parameters': sum(p.numel() for p in model.parameters()), 'raw_rows': raw, 'clean_rows': len(df), 'split_counts': {n: len(f) for n,f in [('train',train_df),('validation',val_df),('test',test_df)]}, 'torch_version': str(torch.__version__)}
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    for name, frame in [('train',train_df),('validation',val_df),('test',test_df)]:
        frame[['label','text']].to_csv(OUT / f'{name}_split.csv', index=False)
    df.label.value_counts().plot.bar(title='Class distribution after duplicate removal')
    plt.ylabel('Messages'); plt.tight_layout(); plt.savefig(OUT / 'class_distribution.png'); plt.close()
    plt.plot([h['epoch'] for h in history], [h['training_loss'] for h in history], label='Training')
    plt.plot([h['epoch'] for h in history], [h['validation_loss'] for h in history], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Weighted binary cross-entropy'); plt.legend(); plt.tight_layout(); plt.savefig(OUT / 'learning_curve.png'); plt.close()
    print(f'Saved model, metrics, charts and splits in {OUT}')


def predict():
    path = OUT / 'model.pt'
    if not path.exists():
        raise SystemExit('Train first: python project.py train')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = SpamLSTM(len(checkpoint['vocab']))
    model.load_state_dict(checkpoint['state']); model.eval()
    print('Type an SMS; type quit to exit. Scores are not calibrated probabilities.')
    while True:
        message = input('\nSMS: ').strip()
        if message.lower() == 'quit':
            break
        if not message:
            continue
        x, lengths = encode([message], checkpoint['vocab'])
        with torch.no_grad():
            score = torch.sigmoid(model(x, lengths)).item()
        print(f"{'SPAM' if score >= 0.5 else 'HAM (not spam)'} | spam score: {score:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    train(args.epochs) if args.command == 'train' else predict()
