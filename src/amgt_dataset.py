import torch
from torch.utils.data import Dataset
import pandas as pd
import sentencepiece as spm
from transformers import AutoTokenizer
import fugashi
from pathlib import Path

ASPECT_MAP = {'room': 0, 'hospitality': 1, 'location': 2, 'bath': 3, 'facility': 4, 'cleanliness': 5, 'breakfast': 6, 'dinner': 7}

class AMGTDataset(Dataset):
    """
    PyTorch Dataset that loads the aspect-level evaluation CSVs and tokenizes 
    the text into 3 separate granularities, ensuring they are padded/truncated 
    to exactly the same sequence length to allow strict 1-to-1 fusion.
    """
    def __init__(self, csv_path, max_length=128):
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length
        
        # Load Tokenizers
        base_dir = Path(__file__).parent.parent
        sp_path = base_dir / "data" / "processed" / "sp_japanese.model"
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_path))
        self.bert_tok = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
        self.morph_tagger = fugashi.Tagger()
        
        # Vocabulary size constraints matching amgt_model.py
        self.morph_vocab_size = 50000
        self.char_vocab_size = 5000

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        aspect_str = row['aspect']
        text = f"[{aspect_str}] {row['comment']}"
        label = int(row['sentiment_label'])
        aspect_idx = ASPECT_MAP.get(aspect_str, -1)
        
        # ----------------------------------------------------
        # 1. Subword/Contextual Level (BERT)
        # ----------------------------------------------------
        bert_enc = self.bert_tok(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = bert_enc['input_ids'].squeeze(0)
        attention_mask = bert_enc['attention_mask'].squeeze(0)
        
        # ----------------------------------------------------
        # 2. Morphological Level (Fugashi)
        # ----------------------------------------------------
        morphs = [w.surface for w in self.morph_tagger(text)]
        morph_ids = [hash(m) % self.morph_vocab_size for m in morphs]
        
        if len(morph_ids) > self.max_length:
            morph_ids = morph_ids[:self.max_length]
        else:
            morph_ids = morph_ids + [0] * (self.max_length - len(morph_ids))
            
        # ----------------------------------------------------
        # 3. Character Level
        # ----------------------------------------------------
        chars = list(text)
        char_ids = [ord(c) % self.char_vocab_size for c in chars]
        
        if len(char_ids) > self.max_length:
            char_ids = char_ids[:self.max_length]
        else:
            char_ids = char_ids + [0] * (self.max_length - len(char_ids))
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'char_ids': torch.tensor(char_ids, dtype=torch.long),
            'morph_ids': torch.tensor(morph_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'aspect_idx': torch.tensor(aspect_idx, dtype=torch.long)
        }

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AMGT DATASET LOADER")
    print("=" * 60)
    
    val_path = Path(__file__).parent.parent / "data" / "processed" / "hotel_val.csv"
    dataset = AMGTDataset(val_path, max_length=64)
    
    print(f"Loaded {len(dataset)} validation records.")
    
    sample = dataset[0]
    print("\nBatch shapes for first record:")
    print(f"  input_ids:      {sample['input_ids'].shape}")
    print(f"  attention_mask: {sample['attention_mask'].shape}")
    print(f"  char_ids:       {sample['char_ids'].shape}")
    print(f"  morph_ids:      {sample['morph_ids'].shape}")
    print(f"  labels:         {sample['labels']}")
    
    print("\nAll sequence arrays are perfectly aligned to max_length=64 for fusion!")
    print("=" * 60)
