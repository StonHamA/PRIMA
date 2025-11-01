import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPImageProcessor, get_cosine_schedule_with_warmup

from prima_framework import (
    CrossModalTransformer,
    MedCLIPVisionEncoder,
    ModalityDiscriminator,
    MultimodalKnowledgeIndexer,
    PRIMAObjective,
    PrototypeInitConfig,
    PubMedBERTTextEncoder,
    initialize_prototypes,
)


@dataclass
class TrainingConfig:
    train_manifest: str
    prototype_manifest: str
    output_dir: str
    vision_model: str = ""
    text_model: str = ""
    max_text_length: int = 96
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    num_workers: int = 4
    device: str = "cuda"
    temperature: float = 0.07
    alpha: float = 0.7
    beta: float = 0.2
    lambda_diversity: float = 0.1
    prototype_k: int = 80
    prototype_top_r: int = 8


class IRMultimodalDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_processor: CLIPImageProcessor,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            self.samples = [json.loads(line) for line in fp if line.strip()]
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        text = sample["text"]

        image_inputs = self.image_processor(images=image, return_tensors="pt")
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }


def load_prototype_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def compute_prototype_seed_embeddings(
    manifest: List[Dict[str, str]],
    vision_encoder: MedCLIPVisionEncoder,
    text_encoder: PubMedBERTTextEncoder,
    image_processor: CLIPImageProcessor,
    tokenizer: AutoTokenizer,
    device: torch.device,
    top_r: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keyword_embeddings: List[torch.Tensor] = []
    image_embeddings: List[torch.Tensor] = []

    vision_encoder.eval()
    text_encoder.eval()

    for entry in manifest:
        keyword = entry["keyword"]
        images = entry.get("image_paths", [])[:top_r]
        if len(images) < top_r and len(images) > 0:
            images = images + images[: max(0, top_r - len(images))]
        elif len(images) == 0:
            continue

        text_inputs = tokenizer(
            keyword,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            token_features = text_encoder(**text_inputs)
            keyword_embedding = token_features.mean(dim=1).squeeze(0).cpu()
        keyword_embeddings.append(keyword_embedding)

        for image_path in images[:top_r]:
            image = Image.open(image_path).convert("RGB")
            image_inputs = image_processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            with torch.no_grad():
                patch_features = vision_encoder(**image_inputs)
                pooled = patch_features.mean(dim=1).squeeze(0).cpu()
            image_embeddings.append(pooled)

    if not keyword_embeddings or not image_embeddings:
        raise ValueError("Prototype manifest did not yield any embeddings. Check paths and keywords.")

    keyword_tensor = torch.stack(keyword_embeddings, dim=0)
    image_tensor = torch.stack(image_embeddings, dim=0)
    return keyword_tensor, image_tensor


def create_repository(
    cfg: TrainingConfig,
    vision_encoder: MedCLIPVisionEncoder,
    text_encoder: PubMedBERTTextEncoder,
    image_processor: CLIPImageProcessor,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.nn.Module:
    manifest = load_prototype_manifest(cfg.prototype_manifest)
    keyword_embs, image_embs = compute_prototype_seed_embeddings(
        manifest,
        vision_encoder,
        text_encoder,
        image_processor,
        tokenizer,
        device,
        cfg.prototype_top_r,
    )
    init_cfg = PrototypeInitConfig(
        keywords=[m["keyword"] for m in manifest],
        keyword_embeddings=keyword_embs,
        image_embeddings=image_embs,
        k=cfg.prototype_k,
        top_r=cfg.prototype_top_r,
    )
    repository = initialize_prototypes(init_cfg)
    return repository


def train(cfg: TrainingConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_model)
    vision_encoder = MedCLIPVisionEncoder(cfg.vision_model).to(device)
    text_encoder = PubMedBERTTextEncoder(cfg.text_model).to(device)

    assert (
        vision_encoder.hidden_size == text_encoder.hidden_size
    ), "Vision and text encoders must share hidden size for alignment."
    alignment_dim = vision_encoder.hidden_size

    repository = create_repository(cfg, vision_encoder, text_encoder, image_processor, tokenizer, device).to(device)
    vision_encoder.train()
    text_encoder.train()

    dataset = IRMultimodalDataset(cfg.train_manifest, image_processor, tokenizer, cfg.max_text_length)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    alignment_module = CrossModalTransformer(dim=alignment_dim, num_heads=8, depth=2)
    discriminator = ModalityDiscriminator(dim=alignment_dim)

    indexer = MultimodalKnowledgeIndexer(
        visual_encoder=vision_encoder,
        text_encoder=text_encoder,
        alignment=alignment_module,
        discriminator=discriminator,
        alignment_dim=alignment_dim,
        projection_dim=alignment_dim,
    ).to(device)

    objective = PRIMAObjective(
        repository=repository,
        temperature=cfg.temperature,
        lambda_modality=cfg.alpha,
        lambda_diversity=cfg.lambda_diversity,
        lambda_prototype=cfg.beta,
        prototype_temperature=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(indexer.parameters()) + list(objective.repository.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(dataloader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(cfg.warmup_steps, total_steps // 10),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(cfg.epochs):
        indexer.train()
        objective.train()
        running = {"total": 0.0, "contrast": 0.0, "modality": 0.0, "prototype": 0.0}
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for batch in progress:
            image_inputs = {"pixel_values": batch["pixel_values"].to(device, non_blocking=True)}
            text_inputs = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                embeddings, fused_visual, fused_text, modality_logits, token_lengths = indexer(image_inputs, text_inputs)
                losses = objective(fused_visual, fused_text, modality_logits, token_lengths)

            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["total"] += losses["loss"].item()
            running["contrast"] += losses["loss_contrast"].item()
            running["modality"] += losses["loss_modality"].item()
            running["prototype"] += losses["loss_prototype"].item()

            step = progress.n
            if step > 0:
                progress.set_postfix(
                    loss=running["total"] / step,
                    contrast=running["contrast"] / step,
                    modality=running["modality"] / step,
                    prototype=running["prototype"] / step,
                )

        torch.save(
            {
                "epoch": epoch + 1,
                "indexer": indexer.state_dict(),
                "repository": repository.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg.__dict__,
            },
            os.path.join(cfg.output_dir, f"prima_epoch_{epoch + 1}.pt"),
        )


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train PRIMA")
    parser.add_argument("--train-manifest", required=True, help="Path to JSONL training manifest.")
    parser.add_argument("--prototype-manifest", required=True, help="Path to JSON prototype manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints.")
    parser.add_argument("--vision-model", default="openmedlab/MedCLIP-ViT-B-32")
    parser.add_argument("--text-model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--lambda-diversity", type=float, default=0.1)
    parser.add_argument("--prototype-k", type=int, default=80)
    parser.add_argument("--prototype-top-r", type=int, default=8)
    args = parser.parse_args()
    return TrainingConfig(
        train_manifest=args.train_manifest,
        prototype_manifest=args.prototype_manifest,
        output_dir=args.output_dir,
        vision_model=args.vision_model,
        text_model=args.text_model,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        device=args.device,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        lambda_diversity=args.lambda_diversity,
        prototype_k=args.prototype_k,
        prototype_top_r=args.prototype_top_r,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
