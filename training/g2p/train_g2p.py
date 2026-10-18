"""Train the English grapheme-to-phoneme fallback used for words missing from
the CMU dictionary (product names, people's names).

The model is trained only on the BSD-licensed CMU dictionary that the native
frontend already ships, so the weights carry no other licence. It is a small
attention seq2seq: a bidirectional GRU over letters and a GRU decoder emitting
ARPAbet phones with stress digits, decoded greedily. The architecture is kept
deliberately plain so the Rust frontend (crates/tts-core/src/g2p.rs) can run it
natively, without ONNX Runtime, and reproduce these exact outputs.

Usage: .venv/bin/python training/g2p/train_g2p.py voices/kokoro-en-q8/cmudict.dict assets/g2p-en-v1.bin
"""

import argparse
import hashlib
import json
import random
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

LETTERS = list("abcdefghijklmnopqrstuvwxyz'")
EMB, ENC, DEC = 64, 192, 384  # DEC = 2 * ENC: the bidirectional states are its context
MAX_PHONES = 32


def load(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(";;;"):
            continue
        parts = line.split("#")[0].split()
        if len(parts) < 2 or "(" in parts[0]:
            continue
        word = parts[0].lower()
        if not all(c in LETTERS for c in word) or len(word) > 30:
            continue
        entries.setdefault(word, parts[1:])
    return entries


def split(word):
    """Stable train/dev/test assignment by hash: 95/2.5/2.5."""
    h = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 1000
    return "test" if h < 25 else "dev" if h < 50 else "train"


class G2P(nn.Module):
    def __init__(self, phones):
        super().__init__()
        self.letters = nn.Embedding(len(LETTERS) + 1, EMB)  # 0 = padding
        self.encoder = nn.GRU(EMB, ENC, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * ENC, DEC)
        self.phones = nn.Embedding(len(phones) + 2, EMB)  # 0 pad, 1 start/end
        self.decoder = nn.GRUCell(EMB + DEC, DEC)
        self.out = nn.Linear(2 * DEC, len(phones) + 2)

    def encode(self, letters, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.letters(letters), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        states, _ = self.encoder(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True)
        mask = letters[:, : states.size(1)] != 0
        mean = (states * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        return states, mask, torch.tanh(self.bridge(mean))

    def step(self, prev, hidden, context, states, mask):
        hidden = self.decoder(torch.cat([self.phones(prev), context], -1), hidden)
        scores = torch.bmm(states, hidden.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        context = torch.bmm(torch.softmax(scores, -1).unsqueeze(1), states).squeeze(1)
        return self.out(torch.cat([hidden, context], -1)), hidden, context

    def forward(self, letters, lengths, targets):
        states, mask, hidden = self.encode(letters, lengths)
        context = torch.zeros(letters.size(0), DEC, device=letters.device)
        prev = torch.ones(letters.size(0), dtype=torch.long, device=letters.device)
        logits = []
        for t in range(targets.size(1)):
            step, hidden, context = self.step(prev, hidden, context, states, mask)
            logits.append(step)
            prev = targets[:, t]
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, letters, lengths):
        states, mask, hidden = self.encode(letters, lengths)
        context = torch.zeros(letters.size(0), DEC, device=letters.device)
        prev = torch.ones(letters.size(0), dtype=torch.long, device=letters.device)
        out = []
        for _ in range(MAX_PHONES):
            step, hidden, context = self.step(prev, hidden, context, states, mask)
            prev = step.argmax(-1)
            out.append(prev)
        return torch.stack(out, 1)


def batch(items, phone_ids, device):
    words, prons = zip(*items)
    lengths = torch.tensor([len(w) for w in words])
    letters = torch.zeros(len(words), int(lengths.max()), dtype=torch.long)
    for i, w in enumerate(words):
        letters[i, : len(w)] = torch.tensor([LETTERS.index(c) + 1 for c in w])
    target_len = max(len(p) for p in prons) + 1
    targets = torch.zeros(len(words), target_len, dtype=torch.long)
    for i, p in enumerate(prons):
        ids = [phone_ids[x] for x in p] + [1]
        targets[i, : len(ids)] = torch.tensor(ids)
    return letters.to(device), lengths.to(device), targets.to(device)


def decode(ids, phones):
    out = []
    for i in ids:
        if i == 1:
            break
        if i >= 2:
            out.append(phones[i - 2])
    return out


def evaluate(model, items, phone_ids, phones, device):
    model.eval()
    words_wrong = phones_wrong = phones_total = 0
    for start in range(0, len(items), 1024):
        chunk = items[start : start + 1024]
        letters, lengths, _ = batch(chunk, phone_ids, device)
        for (word, ref), ids in zip(chunk, model.greedy(letters, lengths).tolist()):
            hyp = decode(ids, phones)
            words_wrong += hyp != ref
            phones_wrong += edit_distance(hyp, ref)
            phones_total += len(ref)
    model.train()
    return words_wrong / len(items), phones_wrong / phones_total


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def export(model, phones, path, metrics):
    """Little-endian f32 tensors after a JSON header: {"phones", "letters",
    "tensors": [[name, shape]...], ...}, prefixed by its u32 byte length."""
    tensors = []
    blob = bytearray()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().float().contiguous()
        tensors.append([name, list(array.shape)])
        blob += struct.pack(f"<{array.numel()}f", *array.flatten().tolist())
    header = json.dumps(
        {
            "format": "local-tts-g2p-v1",
            "letters": "".join(LETTERS),
            "phones": phones,
            "dims": {"emb": EMB, "enc": ENC, "dec": DEC, "max_phones": MAX_PHONES},
            "tensors": tensors,
            "metrics": metrics,
            "training": "CMU Pronouncing Dictionary (BSD) only; see training/g2p/train_g2p.py",
        },
        separators=(",", ":"),
    ).encode()
    Path(path).write_bytes(struct.pack("<I", len(header)) + header + bytes(blob))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dictionary")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    entries = load(args.dictionary)
    phones = sorted({p for pron in entries.values() for p in pron})
    phone_ids = {p: i + 2 for i, p in enumerate(phones)}
    sets = {"train": [], "dev": [], "test": []}
    for word, pron in entries.items():
        if len(pron) < MAX_PHONES:
            sets[split(word)].append((word, pron))
    print({k: len(v) for k, v in sets.items()}, len(phones), "phones", device, flush=True)

    model = G2P(phones).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    best = (1.0, None)
    for epoch in range(args.epochs):
        random.shuffle(sets["train"])
        began = time.time()
        for start in range(0, len(sets["train"]), 256):
            letters, lengths, targets = batch(sets["train"][start : start + 256], phone_ids, device)
            logits = model(letters, lengths, targets)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        schedule.step()
        wer, per = evaluate(model, sets["dev"], phone_ids, phones, device)
        print(f"epoch {epoch + 1}: loss {loss.item():.3f} dev WER {wer:.3f} PER {per:.4f} ({time.time() - began:.0f}s)", flush=True)
        if wer < best[0]:
            best = (wer, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    wer, per = evaluate(model, sets["test"], phone_ids, phones, device)
    print(f"test WER {wer:.3f} PER {per:.4f}", flush=True)
    model = model.cpu()
    export(model, phones, args.output, {"test_word_error": round(wer, 4), "test_phone_error": round(per, 4)})
    # Reference outputs for the Rust parity test.
    probe = [w for w, _ in sets["test"][:200]] + ["figma", "canva", "shopify", "kubernetes", "oleksandr", "tiktok"]
    letters, lengths, _ = batch([(w, ["AA0"]) for w in probe], phone_ids, "cpu")
    reference = {w: " ".join(decode(ids, phones)) for w, ids in zip(probe, model.greedy(letters, lengths).tolist())}
    Path(args.output).with_suffix(".reference.json").write_text(json.dumps(reference, indent=1))
    print("wrote", args.output, Path(args.output).stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    sys.exit(main())
