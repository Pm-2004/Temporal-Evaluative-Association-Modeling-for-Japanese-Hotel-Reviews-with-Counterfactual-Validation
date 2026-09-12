import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class AdaptiveGranularityFusion(nn.Module):
    """
    Custom fusion layer that projects character, morphological, and subword 
    embeddings into a unified space, and adaptively gates them before
    passing the fused representation to the contextual model.
    """
    def __init__(self, char_dim, morph_dim, subword_dim, hidden_dim):
        super().__init__()
        # Projections to unify dimensions
        self.char_proj = nn.Linear(char_dim, hidden_dim)
        self.morph_proj = nn.Linear(morph_dim, hidden_dim)
        self.subword_proj = nn.Linear(subword_dim, hidden_dim)
        
        # Adaptive gating (learns how much of each granularity to use)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.Sigmoid()
        )
        self.fusion_out = nn.Linear(hidden_dim * 3, hidden_dim)
        
    def forward(self, char_emb, morph_emb, subword_emb):
        # Note: char_emb, morph_emb, subword_emb are expected to be sequence-aligned 
        # or pooled to a fixed aspect-representation length beforehand.
        c = self.char_proj(char_emb)
        m = self.morph_proj(morph_emb)
        s = self.subword_proj(subword_emb)
        
        concat_emb = torch.cat([c, m, s], dim=-1) # Shape: (batch, seq, hidden_dim * 3)
        gated = concat_emb * self.gate(concat_emb)
        
        return self.fusion_out(gated)


class AMGTForAspectEvaluation(nn.Module):
    """
    Adaptive Multi-Granularity Transformer (AMGT) for Aspect-Level Evaluative Association.
    Takes multi-granularity inputs, fuses them, processes them through BERT, 
    and predicts aspect sentiment/evaluation.
    """
    def __init__(self, pretrained_model_name="cl-tohoku/bert-base-japanese-v3", num_labels=3, class_weights=None):
        super().__init__()
        self.num_labels = num_labels
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        
        # 1. Base Contextual Model (BERT)
        # We load without the classification head so we can use the hidden states directly
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # 2. Embedding layers for custom granularities
        # (Vocabulary sizes here are placeholders for the PyTorch Dataset logic)
        self.char_embeddings = nn.Embedding(num_embeddings=5000, embedding_dim=128) 
        self.morph_embeddings = nn.Embedding(num_embeddings=50000, embedding_dim=256)
        
        # 3. Adaptive Fusion Layer
        hidden_dim = self.config.hidden_size
        self.fusion = AdaptiveGranularityFusion(
            char_dim=128,
            morph_dim=256,
            subword_dim=hidden_dim, # We use BERT's native word embeddings as the subword base
            hidden_dim=hidden_dim
        )
        
        # 4. Aspect-Evaluation Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, char_ids, morph_ids, labels=None, aspect_idx=None, **kwargs):
        # 1. Obtain base embeddings for each granularity
        char_emb = self.char_embeddings(char_ids)
        morph_emb = self.morph_embeddings(morph_ids)
        
        # For subwords, we extract the raw embeddings from BERT before the encoder
        subword_emb = self.bert.embeddings.word_embeddings(input_ids)
        
        # 2. Fuse the representations
        # *CRITICAL ENGINEERING NOTE*: char_ids and morph_ids must be aligned 
        # in sequence length to input_ids (using attention pooling) in the Dataset layer
        # before reaching here. Assuming sequence-aligned for this forward pass:
        fused_embeddings = self.fusion(char_emb, morph_emb, subword_emb)
        
        # 3. Pass the fused embeddings directly into BERT using inputs_embeds
        # This bypasses the standard token embedding layer cleanly.
        bert_outputs = self.bert(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask
        )
        
        sequence_output = bert_outputs.last_hidden_state
        
        # 4. Pool the CLS token for the final evaluation sequence classification
        cls_token = sequence_output[:, 0, :]
        
        # Predict evaluation strength (0: Negative, 1: Neutral, 2: Positive)
        logits = self.classifier(cls_token)
        
        # Pass aspect_idx through safely by appending it
        if aspect_idx is not None:
            logits_with_aspect = torch.cat([logits, aspect_idx.unsqueeze(1).to(logits.device)], dim=1)
        else:
            logits_with_aspect = logits
        
        loss = None
        if labels is not None:
            if getattr(self, "_print_weight_once", True):
                print(f"\n[DIAGNOSTIC 2] Class weights device: {self.class_weights.device if self.class_weights is not None else 'None'}, values: {self.class_weights}")
                self._print_weight_once = False
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits_with_aspect)
