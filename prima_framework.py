import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import AutoModel, CLIPVisionModel


class CrossModalTransformer(nn.Module):
    """Cross-modal alignment with multi-head attention and transformer blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(visual_tokens)
        k = self.key_proj(text_tokens)
        v = self.value_proj(text_tokens)
        attn_scores = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1)), dim=-1)
        fused = torch.matmul(attn_scores, v)
        fused = self.output_proj(fused) + visual_tokens
        enriched = self.encoder(fused)
        return enriched


class ModalityDiscriminator(nn.Module):
    """Predicts whether tokens come from visual or textual stream."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(tokens)


class MultimodalKnowledgeIndexer(nn.Module):
    """Dual encoder with anatomy-aware alignment and modality preservation."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        text_encoder: nn.Module,
        alignment: CrossModalTransformer,
        discriminator: ModalityDiscriminator,
        alignment_dim: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.alignment = alignment
        self.discriminator = discriminator
        self.projection = nn.Linear(alignment_dim, projection_dim)

    def forward(
        self,
        image_inputs: Dict[str, torch.Tensor],
        text_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_tokens = self.visual_encoder(**image_inputs)
        text_tokens = self.text_encoder(**text_inputs)
        fused_visual = self.alignment(visual_tokens, text_tokens)
        fused_text = self.alignment(text_tokens, visual_tokens)
        pooled = fused_visual.mean(dim=1)
        embeddings = F.normalize(self.projection(pooled), dim=-1)
        logits_visual = self.discriminator(fused_visual)
        logits_text = self.discriminator(fused_text)
        logits = torch.cat([logits_visual, logits_text], dim=1)
        token_lengths = (fused_visual.size(1), fused_text.size(1))
        return embeddings, fused_visual, fused_text, logits, token_lengths


def spatial_contrastive_loss(
    visual_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    b, n, d = visual_tokens.shape
    visual_flat = visual_tokens.reshape(b * n, d)
    text_flat = text_tokens.reshape(b * n, d)
    visual_flat = F.normalize(visual_flat, dim=-1)
    text_flat = F.normalize(text_flat, dim=-1)
    logits = torch.matmul(visual_flat, text_flat.t()) / temperature
    labels = torch.arange(b * n, device=visual_tokens.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


def modality_preservation_loss(
    logits: torch.Tensor, modality_labels: torch.Tensor
) -> torch.Tensor:
    logits = logits.reshape(-1, logits.size(-1))
    labels = modality_labels.reshape(-1)
    return F.cross_entropy(logits, labels)


@dataclass
class PrototypeInitConfig:
    keywords: List[str]
    keyword_embeddings: torch.Tensor
    image_embeddings: torch.Tensor
    k: int
    top_r: int


class PrototypeRepository(nn.Module):
    """Stores and updates semantic prototypes."""

    def __init__(self, prototypes: torch.Tensor) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(prototypes)

    def soft_assignments(self, embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        distances = torch.cdist(embeddings, self.prototypes, p=2).pow(2)
        return F.softmax(-distances / temperature, dim=-1)

    def regularization_loss(
        self,
        embeddings: torch.Tensor,
        temperature: float,
        lambda_diversity: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.soft_assignments(embeddings, temperature)
        distances = torch.cdist(embeddings, self.prototypes, p=2).pow(2)
        compactness = (weights * distances).sum(dim=-1).mean()
        proto_sim = torch.matmul(self.prototypes, self.prototypes.t())
        mask = ~torch.eye(proto_sim.size(0), dtype=torch.bool, device=proto_sim.device)
        diversity = torch.exp(proto_sim[mask]).mean()
        return compactness + lambda_diversity * diversity, weights


def initialize_prototypes(config: PrototypeInitConfig) -> PrototypeRepository:
    keyword_emb = config.keyword_embeddings
    image_emb = config.image_embeddings
    pooled_image = image_emb.view(keyword_emb.size(0), config.top_r, -1).mean(dim=1)
    fused = F.normalize(keyword_emb + pooled_image, dim=-1)
    kmeans = KMeans(n_clusters=config.k, n_init="auto")
    assignments = kmeans.fit_predict(fused.detach().cpu().numpy())
    centroids = []
    for idx in range(config.k):
        mask = assignments == idx
        cluster = fused[mask]
        if len(cluster) == 0:
            centroids.append(torch.randn_like(fused[0]))
        else:
            centroids.append(cluster.mean(dim=0))
    proto_tensor = torch.stack(centroids, dim=0)
    return PrototypeRepository(proto_tensor)


class PrototypeGuidedRetriever:
    """Reranks retrieved candidates using prototype-weighted scoring."""

    def __init__(self, repository: PrototypeRepository, gamma: float = 1.0) -> None:
        self.repository = repository
        self.gamma = gamma

    def rerank(
        self,
        query_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(query_embedding, dim=-1)
        candidates = F.normalize(candidate_embeddings, dim=-1)
        base_scores = torch.matmul(candidates, query.unsqueeze(-1)).squeeze(-1)
        weights = self.repository.soft_assignments(query.unsqueeze(0)).squeeze(0)
        proto_scores = torch.matmul(candidates, self.repository.prototypes.t())
        weighted = torch.matmul(proto_scores, weights)
        return base_scores + self.gamma * weighted


class PRIMAObjective(nn.Module):
    """Combines contrastive, modality, and prototype objectives."""

    def __init__(
        self,
        repository: PrototypeRepository,
        temperature: float = 0.07,
        lambda_modality: float = 0.1,
        lambda_diversity: float = 0.1,
        lambda_prototype: float = 1.0,
        prototype_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.temperature = temperature
        self.lambda_modality = lambda_modality
        self.lambda_diversity = lambda_diversity
        self.lambda_prototype = lambda_prototype
        self.prototype_temperature = prototype_temperature

    def forward(
        self,
        fused_visual: torch.Tensor,
        fused_text: torch.Tensor,
        modality_logits: torch.Tensor,
        token_lengths: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        loss_contrast = spatial_contrastive_loss(fused_visual, fused_text, self.temperature)
        visual_len, text_len = token_lengths
        bsz = fused_visual.size(0)
        device = fused_visual.device
        visual_labels = torch.zeros(bsz, visual_len, dtype=torch.long, device=device)
        text_labels = torch.ones(bsz, text_len, dtype=torch.long, device=device)
        modality_labels = torch.cat([visual_labels, text_labels], dim=1)
        loss_modality = modality_preservation_loss(modality_logits, modality_labels)
        embeddings = fused_visual.mean(dim=1)
        loss_proto_raw, weights = self.repository.regularization_loss(
            embeddings,
            temperature=self.prototype_temperature,
            lambda_diversity=self.lambda_diversity,
        )
        loss_proto = self.lambda_prototype * loss_proto_raw
        total = loss_contrast + self.lambda_modality * loss_modality + loss_proto
        return {
            "loss": total,
            "loss_contrast": loss_contrast,
            "loss_modality": loss_modality,
            "loss_prototype": loss_proto,
            "prototype_weights": weights.detach(),
        }


def build_dummy_encoder(dim_in: int, dim_out: int) -> nn.Module:
    """Creates a lightweight transformer-compatible encoder for prototyping."""

    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(dim_in, dim_out)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = self.proj(x)
            return x

    return DummyEncoder()


class MedCLIPVisionEncoder(nn.Module):
    """Wrapper around MedCLIP/CLIP vision backbones that returns patch tokens."""

    def __init__(self, model_name: str, trainable: bool = True, output_attentions: bool = False) -> None:
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name, output_attentions=output_attentions)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor, **_) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state


class PubMedBERTTextEncoder(nn.Module):
    """Wrapper around PubMedBERT encoder that returns token representations."""

    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def example_batch_step(
    indexer: MultimodalKnowledgeIndexer,
    objective: PRIMAObjective,
    image_inputs: Dict[str, torch.Tensor],
    text_inputs: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> Dict[str, torch.Tensor]:
    indexer.train()
    optimizer.zero_grad()
    embeddings, fused_visual, fused_text, modality_logits, token_lengths = indexer(image_inputs, text_inputs)
    losses = objective(fused_visual, fused_text, modality_logits, token_lengths)
    losses["loss"].backward()
    optimizer.step()
    losses["embeddings"] = embeddings.detach()
    return losses
