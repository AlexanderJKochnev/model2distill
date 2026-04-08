#!/usr/bin/env python3
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from model2vec.distill import distill
from model2vec import StaticModel

MODELS_DIR = Path("/models")           # том для сохранения модели
CACHE_DIR = Path("/cache/huggingface") # том для кэша Hugging Face (опционально)
DISTILLED_PATH = MODELS_DIR / "distilled_e5_256d"
ORIGINAL_MODEL_NAME = "intfloat/multilingual-e5-small"
PCA_DIMS = 256

def main():
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    if DISTILLED_PATH.exists():
        print(f"✅ Distilled model already exists at {DISTILLED_PATH}")
        return

    print("📥 Downloading original model...")
    # Указываем кэш-директорию, чтобы не засорять корень
    original_model = SentenceTransformer(ORIGINAL_MODEL_NAME, device="cpu", cache_folder=str(CACHE_DIR))

    print("⚙️ Distilling to 256 dims...")
    m2v_model = distill(
        model_name=ORIGINAL_MODEL_NAME,
        pca_dims=PCA_DIMS,
    )

    print(f"💾 Saving to {DISTILLED_PATH}")
    m2v_model.save_pretrained(str(DISTILLED_PATH))

    # Проверка
    test_vec = m2v_model.encode(["test"])[0]
    print(f"✅ Done. Vector dim: {len(test_vec)}")

if __name__ == "__main__":
    main()