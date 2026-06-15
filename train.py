import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.transformer import Transformer


# ── Dataset ────────────────────────────────────────────────────────────────────

PAIRS = [
    ("il gatto dorme", "the cat sleeps"),
    ("il cane mangia", "the dog eats"),
    ("la ragazza corre", "the girl runs"),
    ("il bambino piange", "the child cries"),
    ("la donna legge", "the woman reads"),
    ("il sole splende", "the sun shines"),
    ("il gatto mangia", "the cat eats"),
    ("il cane dorme", "the dog sleeps"),
    ("la ragazza legge", "the girl reads"),
    ("il bambino corre", "the child runs"),
]

# Special tokens
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"


def build_vocab(sentences: list[str]) -> dict[str, int]:
    """Build a vocabulary from a list of sentences."""
    vocab = {PAD: 0, SOS: 1, EOS: 2}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def encode(sentence: str, vocab: dict[str, int]) -> list[int]:
    """Encode a sentence into a list of token indices."""
    return [vocab[SOS]] + [vocab[w] for w in sentence.split()] + [vocab[EOS]]


def pad_sequence(seq: list[int], max_len: int, pad_idx: int = 0) -> list[int]:
    """Pad a sequence to max_len."""
    return seq + [pad_idx] * (max_len - len(seq))


class TranslationDataset(Dataset):
    """Simple Italian → English translation dataset."""

    def __init__(self, pairs: list[tuple[str, str]]):
        src_sentences = [p[0] for p in pairs]
        tgt_sentences = [p[1] for p in pairs]

        self.src_vocab = build_vocab(src_sentences)
        self.tgt_vocab = build_vocab(tgt_sentences)
        self.tgt_idx_to_word = {v: k for k, v in self.tgt_vocab.items()}

        # Encode all sentences
        src_encoded = [encode(s, self.src_vocab) for s in src_sentences]
        tgt_encoded = [encode(s, self.tgt_vocab) for s in tgt_sentences]

        # Pad to max length
        src_max_len = max(len(s) for s in src_encoded)
        tgt_max_len = max(len(s) for s in tgt_encoded)

        self.src = torch.tensor([pad_sequence(s, src_max_len) for s in src_encoded])
        self.tgt = torch.tensor([pad_sequence(s, tgt_max_len) for s in tgt_encoded])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


# ── Training ───────────────────────────────────────────────────────────────────

def train(num_epochs: int = 300):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and dataloader
    dataset = TranslationDataset(PAIRS)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    model = Transformer(
        src_vocab_size=len(dataset.src_vocab),
        tgt_vocab_size=len(dataset.tgt_vocab),
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            # Forward pass
            logits, _ = model(src, tgt_input)

            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_target.reshape(-1),
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    print("\nTraining complete!")
    torch.save(model.state_dict(), "transformer.pth")
    print("Model saved to transformer.pth")
    return model, dataset


# ── Inference ──────────────────────────────────────────────────────────────────

def translate(
    model: Transformer,
    sentence: str,
    dataset: TranslationDataset,
    max_len: int = 10,
) -> str:
    """Translate a sentence using greedy decoding."""
    device = next(model.parameters()).device
    model.eval()

    # Encode source
    src_encoded = encode(sentence, dataset.src_vocab)
    src = torch.tensor(src_encoded).unsqueeze(0).to(device)

    # Encode with transformer encoder
    encoder_output, src_mask, _ = model.encode(src)

    # Start decoding with <sos>
    sos_idx = dataset.tgt_vocab[SOS]
    eos_idx = dataset.tgt_vocab[EOS]
    tgt = torch.tensor([[sos_idx]]).to(device)

    result = []
    for _ in range(max_len):
        decoder_output, _, _ = model.decode(tgt, encoder_output, src_mask)
        logits = model.output_projection(decoder_output[:, -1, :])
        next_token = logits.argmax(dim=-1).item()

        if next_token == eos_idx:
            break

        result.append(dataset.tgt_idx_to_word[next_token])
        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)

    return " ".join(result)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, dataset = train()

    print("\n-- Translations --")
    test_sentences = [
        "il gatto dorme",
        "il cane mangia",
        "la ragazza corre",
        "il bambino piange",
    ]
    for sentence in test_sentences:
        translation = translate(model, sentence, dataset)
        print(f"  {sentence} -> {translation}")
