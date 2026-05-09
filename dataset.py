import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import spacy

from config import UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX, SPECIAL_TOKENS

class Vocabulary:
    """Simple token <-> index mapping."""

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        for idx, tok in enumerate(SPECIAL_TOKENS):
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

    def build_from_sentences(self, tokenized_sentences):
        """
        Add all tokens from a list of token lists to the vocabulary.
        Tokens that already exist (e.g. special tokens) are skipped.
        """
        for tokens in tokenized_sentences:
            for tok in tokens:
                if tok not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[tok] = idx
                    self.idx2token[idx] = tok

    def __len__(self):
        return len(self.token2idx)

    def lookup_token(self, idx: int) -> str:
        return self.idx2token.get(idx, "<unk>")

    def encode(self, tokens) -> list:
        """Convert token list to index list, using UNK for unknowns."""
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]


class Multi30kDataset(Dataset):
    """
    PyTorch Dataset wrapping a single split of Multi30k.

    Args:
        split     : 'train', 'validation', or 'test'.
        src_vocab : Vocabulary for German (source).
        tgt_vocab : Vocabulary for English (target).
        de_nlp    : spaCy German tokenizer.
        en_nlp    : spaCy English tokenizer.
        max_src_len : Max source sequence length (tokens, incl. SOS/EOS).
        max_tgt_len : Max target sequence length (tokens, incl. SOS/EOS).
    """

    def __init__(self, split, src_vocab, tgt_vocab, de_nlp, en_nlp,
                 max_src_len=100, max_tgt_len=100):
        self.src_vocab   = src_vocab
        self.tgt_vocab   = tgt_vocab
        self.de_nlp      = de_nlp
        self.en_nlp      = en_nlp
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        from datasets import load_dataset
        raw = load_dataset("bentrevett/multi30k", split=split)
        self.data = self._process(raw)

    def _tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def _tokenize_en(self, text: str):
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    def _numericalize(self, tokens, vocab, max_len):
        """Encode tokens with SOS/EOS, clipped to max_len."""
        ids = [SOS_IDX] + vocab.encode(tokens) + [EOS_IDX]
        return ids[:max_len]

    def _process(self, raw_dataset):
        processed = []
        for example in raw_dataset:
            de_tokens = self._tokenize_de(example["de"])
            en_tokens = self._tokenize_en(example["en"])
            src_ids = self._numericalize(de_tokens, self.src_vocab, self.max_src_len)
            tgt_ids = self._numericalize(en_tokens, self.tgt_vocab, self.max_tgt_len)
            processed.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Pad sequences within a batch to the same length.
    Returns: (src_batch, tgt_batch) each shape [batch, max_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def build_vocabs(de_nlp, en_nlp):
    """
    Build src (German) and tgt (English) vocabularies from the training set only.
    Test/val sets must NOT influence the vocabulary.

    Returns:
        src_vocab, tgt_vocab : Vocabulary objects
    """
    from datasets import load_dataset
    print("Loading training data for vocabulary construction...")
    train_raw = load_dataset("bentrevett/multi30k", split="train")

    de_sentences = []
    en_sentences = []
    for example in train_raw:
        de_sentences.append([tok.text.lower() for tok in de_nlp.tokenizer(example["de"])])
        en_sentences.append([tok.text.lower() for tok in en_nlp.tokenizer(example["en"])])

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_from_sentences(de_sentences)
    tgt_vocab.build_from_sentences(en_sentences)

    print(f"Source vocab size (de): {len(src_vocab)}")
    print(f"Target vocab size (en): {len(tgt_vocab)}")
    return src_vocab, tgt_vocab


def get_dataloaders(batch_size=128, max_src_len=100, max_tgt_len=100,
                   num_workers=0):
    """
    Full data pipeline: load spaCy models, build vocabs, create DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    print("Loading spaCy tokenizers...")
    try:
        de_nlp = spacy.load("de_core_news_sm")
    except OSError:
        raise OSError(
            "German spaCy model not found. Run:\n"
            "  python -m spacy download de_core_news_sm"
        )
    try:
        en_nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "English spaCy model not found. Run:\n"
            "  python -m spacy download en_core_web_sm"
        )

    src_vocab, tgt_vocab = build_vocabs(de_nlp, en_nlp)

    print("Building datasets...")
    train_ds = Multi30kDataset("train",      src_vocab, tgt_vocab, de_nlp, en_nlp,
                               max_src_len, max_tgt_len)
    val_ds   = Multi30kDataset("validation", src_vocab, tgt_vocab, de_nlp, en_nlp,
                               max_src_len, max_tgt_len)
    test_ds  = Multi30kDataset("test",       src_vocab, tgt_vocab, de_nlp, en_nlp,
                               max_src_len, max_tgt_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
