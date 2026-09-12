import sentencepiece as spm
from pathlib import Path

def main():
    text = "眺めが良く清潔な部屋と温かいお祝いに感激"
    print(f"Original Text: {text}\n")
    print("=" * 50)
    
    # 1. Character Level
    print("1. CHARACTER-LEVEL")
    chars = list(text)
    print(chars)
    print("-" * 50)
    
    # 2. Subword Level (Custom SentencePiece)
    print("2. SUBWORD-LEVEL (Custom SentencePiece)")
    base_dir = Path(__file__).parent.parent
    sp_model_path = base_dir / "data" / "processed" / "sp_japanese.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    print(sp.encode_as_pieces(text))
    print("-" * 50)
    
    # 3. Morphological Level (Fugashi/MeCab)
    print("3. MORPHOLOGICAL-LEVEL (Fugashi)")
    try:
        import fugashi
        tagger = fugashi.Tagger()
        morphs = [word.surface for word in tagger(text)]
        print(morphs)
    except ImportError:
        print("ERROR: 'fugashi' library is not installed. We need this for morphological tokenization.")
    except RuntimeError as e:
        print(f"ERROR initializing Fugashi (you might need unidic-lite): {e}")
    print("-" * 50)
        
    # 4. Contextual Level (Hugging Face BERT)
    print("4. CONTEXTUAL-LEVEL (Hugging Face BERT)")
    try:
        from transformers import AutoTokenizer
        bert_tok = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
        print(bert_tok.tokenize(text))
    except ImportError:
        print("ERROR: 'transformers' library is not installed.")

    print("=" * 50)

if __name__ == "__main__":
    main()
