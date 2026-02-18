!pip install faiss-cpu




"""
Task 3: Semantic Image Retrieval System
Content-Based Image Retrieval (CBIR) for Chest X-Rays
Embedding: BiomedCLIP (Microsoft) or CLIP ViT-B/32 (fallback)
Vector Index: FAISS

Kaggle Dataset: /kaggle/input/chest-xray-pneumonia/chest_xray/
"""

# ─────────────────────────────────────────────
#  Install missing packages (run these in separate cells first if needed)
# ─────────────────────────────────────────────
# !pip install faiss-cpu
# !pip install git+https://github.com/openai/CLIP.git     # for CLIP
# !pip install open_clip_torch                             # for BiomedCLIP

import os
import json
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict

import torch
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

# ── Early check for faiss ──
try:
    import faiss
except ImportError:
    print("\n" + "═"*80)
    print("ERROR: faiss is not installed")
    print("Please run one of these in a separate cell and restart the kernel:")
    print("    !pip install faiss-cpu")
    print("    !pip install faiss-gpu     # if you want GPU acceleration")
    print("═"*80 + "\n")
    raise

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
class Config:
    DATA_ROOT   = Path("/kaggle/input/chest-xray-pneumonia/chest_xray")
    OUTPUT_DIR  = Path("/kaggle/working/task3_outputs")
    INDEX_DIR   = Path("/kaggle/working/task3_outputs/index")

    EMBED_MODEL = "clip"           # "clip" | "biomed_clip" | "resnet"
    IMAGE_SIZE  = 224
    EMBED_DIM   = 512              # CLIP & BiomedCLIP = 512
    TOP_K       = [1, 3, 5, 10]
    SEED        = 42
    CLASSES     = ['NORMAL', 'PNEUMONIA']
    FAISS_TYPE  = "flat"           # "flat" = exact, "ivf" = approximate


def setup_dirs():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
def load_all_images(split: str) -> List[Dict]:
    records = []
    for lbl_idx, cls in enumerate(Config.CLASSES):
        cls_dir = Config.DATA_ROOT / split / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.jpeg")):
            records.append({'path': p, 'label': cls, 'label_idx': lbl_idx})
    print(f"[{split.upper()}] Loaded {len(records)} images")
    return records


# ─────────────────────────────────────────────
#  EMBEDDERS
# ─────────────────────────────────────────────
class CLIPEmbedder:
    def __init__(self, device):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        print("CLIP ViT-B/32 loaded")

    def encode_image(self, images: List[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch).float()
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def encode_text(self, texts: List[str]) -> np.ndarray:
        import clip
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens).float()
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


class BiomedCLIPEmbedder:
    def __init__(self, device):
        import open_clip
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()
        self.device = device
        print("BiomedCLIP loaded")

    def encode_image(self, images: List[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def get_embedder(device):
    if Config.EMBED_MODEL == "biomed_clip":
        try:
            return BiomedCLIPEmbedder(device)
        except Exception as e:
            print(f"BiomedCLIP load failed: {e}\nFalling back to CLIP")
    try:
        return CLIPEmbedder(device)
    except Exception as e:
        print(f"CLIP load failed: {e}\nFalling back to ResNet (no text support)")
        from torchvision.models import resnet50
        class ResNetFallback:
            def __init__(self, device):
                self.device = device
                model = resnet50(weights='IMAGENET1K_V1')
                self.model = torch.nn.Sequential(*list(model.children())[:-1]).eval().to(device)
                self.preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
            def encode_image(self, images):
                batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
                with torch.no_grad():
                    feats = self.model(batch).squeeze(-1).squeeze(-1)
                    feats /= feats.norm(dim=-1, keepdim=True)
                return feats.cpu().numpy().astype(np.float32)
            def encode_text(self, texts):
                raise NotImplementedError("ResNet fallback has no text encoder")
        return ResNetFallback(device)


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION & INDEX
# ─────────────────────────────────────────────
def extract_embeddings(records, embedder, batch_size=64):
    all_emb = []
    paths, labels = [], []
    for i in tqdm(range(0, len(records), batch_size), desc="Extracting"):
        batch = records[i:i+batch_size]
        imgs = [Image.open(r['path']).convert("RGB") for r in batch]
        embs = embedder.encode_image(imgs)
        all_emb.append(embs)
        paths.extend([str(r['path']) for r in batch])
        labels.extend([r['label_idx'] for r in batch])
    embeddings = np.concatenate(all_emb).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, paths, labels


def save_index_data(embeddings, paths, labels, prefix="test"):
    np.save(Config.INDEX_DIR / f"{prefix}_embeddings.npy", embeddings)
    with open(Config.INDEX_DIR / f"{prefix}_metadata.json", "w") as f:
        json.dump({'paths': paths, 'labels': labels}, f)


def load_index_data(prefix="test"):
    emb = np.load(Config.INDEX_DIR / f"{prefix}_embeddings.npy")
    with open(Config.INDEX_DIR / f"{prefix}_metadata.json") as f:
        meta = json.load(f)
    return emb, meta['paths'], meta['labels']


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    if Config.FAISS_TYPE == "ivf" and len(embeddings) > 1000:
        nlist = min(100, len(embeddings)//10)
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 10
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors")
    return index


def save_faiss_index(index, name="test_index.faiss"):
    faiss.write_index(index, str(Config.INDEX_DIR / name))


def load_faiss_index(name="test_index.faiss"):
    idx = faiss.read_index(str(Config.INDEX_DIR / name))
    print(f"FAISS index loaded: {idx.ntotal} vectors")
    return idx


# ─────────────────────────────────────────────
#  RETRIEVAL SYSTEM
# ─────────────────────────────────────────────
class RetrievalSystem:
    def __init__(self, index, embedder, db_paths, db_labels):
        self.index = index
        self.embedder = embedder
        self.db_paths = db_paths
        self.db_labels = db_labels

    def image_to_image(self, query_path: str, k: int = 5) -> List[Dict]:
        img = Image.open(query_path).convert("RGB")
        q_emb = self.embedder.encode_image([img]).astype(np.float32)
        scores, indices = self.index.search(q_emb, k + 1)
        results = []
        for sc, idx in zip(scores[0], indices[0]):
            if self.db_paths[idx] == query_path:
                continue
            results.append({
                'path': self.db_paths[idx],
                'label': Config.CLASSES[self.db_labels[idx]],
                'score': float(sc),
            })
        return results[:k]

    def text_to_image(self, query_text: str, k: int = 5) -> List[Dict]:
        if not hasattr(self.embedder, 'encode_text'):
            raise AttributeError("Embedder does not support text queries.")
        q_emb = self.embedder.encode_text([query_text]).astype(np.float32)
        scores, indices = self.index.search(q_emb, k)
        return [{
            'path': self.db_paths[i],
            'label': Config.CLASSES[self.db_labels[i]],
            'score': float(s),
        } for s, i in zip(scores[0], indices[0])]


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
def evaluate_precision_at_k(rs: RetrievalSystem, queries: List[Dict], k_values: List[int]):
    results = {k: [] for k in k_values}
    maxk = max(k_values)
    for rec in tqdm(queries, desc="P@k eval"):
        lbl = rec['label_idx']
        ret = rs.image_to_image(str(rec['path']), maxk)
        for kk in k_values:
            top = ret[:kk]
            correct = sum(1 for x in top if Config.CLASSES.index(x['label']) == lbl)
            results[kk].append(correct / kk)
    df = pd.DataFrame({
        f'P@{k}': [np.mean(results[k]), np.std(results[k])]
        for k in k_values
    }, index=['mean', 'std'])
    print("\nPrecision@k:\n", df.round(4))
    df.to_csv(Config.OUTPUT_DIR / "precision_at_k.csv")
    return df


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main(mode="full", query=None, k=5):
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    setup_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {mode}")

    embedder = get_embedder(device)

    if mode in ("search_image", "search_text") and query:
        emb, paths, lbls = load_index_data("test")
        idx = load_faiss_index()
        rs = RetrievalSystem(idx, embedder, paths, lbls)
        if mode == "search_image":
            print(f"\nImage search → {query}")
            for i, r in enumerate(rs.image_to_image(query, k), 1):
                print(f"  #{i} | {r['label']:<10} | score={r['score']:.4f} | {r['path']}")
        else:
            print(f"\nText search → {query}")
            for i, r in enumerate(rs.text_to_image(query, k), 1):
                print(f"  #{i} | {r['label']:<10} | score={r['score']:.4f} | {r['path']}")
        return

    print("\nLoading test set …")
    test_records = load_all_images("test")

    print("\nExtracting embeddings …")
    embeddings, paths, labels = extract_embeddings(test_records, embedder)
    save_index_data(embeddings, paths, labels, "test")

    print("\nBuilding FAISS index …")
    index = build_faiss_index(embeddings)
    save_faiss_index(index)

    rs = RetrievalSystem(index, embedder, paths, labels)

    if mode in ("evaluate", "full"):
        print("\nEvaluating …")
        evaluate_precision_at_k(rs, test_records, Config.TOP_K)

    print("\nTask 3 finished.")
    print(f"Outputs → {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: CBIR for Chest X-rays")
    parser.add_argument("--mode", default="full",
                        choices=["full", "build", "evaluate", "search_image", "search_text"])
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--k", default=5, type=int)

    # ── This line fixes the Jupyter/Kaggle -f kernel.json error ──
    args, unknown = parser.parse_known_args()

    if unknown:
        print("Ignored extra arguments (normal in Jupyter/Kaggle):", unknown)

    main(mode=args.mode, query=args.query, k=args.k)