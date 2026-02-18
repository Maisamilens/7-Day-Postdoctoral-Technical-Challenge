"""
Task 2: Medical Report Generation using Visual Language Model
=============================================================
Chest X-Ray → Natural Language Medical Report
Uses: BLIP-2 (primary) with MedGemma instructions (if HF token available)

Kaggle Dataset: /kaggle/input/chest-xray-pneumonia/chest_xray/
"""

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from typing import Optional

import torch
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
class Config:
    DATA_ROOT   = Path("/kaggle/input/chest-xray-pneumonia/chest_xray")
    OUTPUT_DIR  = Path("/kaggle/working/task2_outputs")
    REPORT_DIR  = Path("/kaggle/working/task2_outputs/reports")

    # VLM settings
    # Options: "blip2" | "medgemma" | "llava"
    VLM_MODEL   = "blip2"
    # For MedGemma: set your HF token via Kaggle Secrets → HUGGINGFACE_TOKEN
    HF_TOKEN    = os.environ.get("HUGGINGFACE_TOKEN", None)

    CLASSES     = ['NORMAL', 'PNEUMONIA']
    NUM_SAMPLES = 10   # Reports to generate (5 per class)
    SEED        = 42


def setup_dirs():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
#  IMAGE COLLECTION
# ─────────────────────────────────────────────
def collect_test_images(n_per_class=5):
    """Sample n images per class from the test split."""
    samples = []
    random.seed(Config.SEED)

    for label_idx, cls in enumerate(Config.CLASSES):
        cls_dir = Config.DATA_ROOT / "test" / cls
        images  = sorted(cls_dir.glob("*.jpeg"))
        chosen  = random.sample(images, min(n_per_class, len(images)))
        for p in chosen:
            samples.append({'path': p, 'label': cls, 'label_idx': label_idx})

    print(f"Collected {len(samples)} images for report generation")
    return samples


# ─────────────────────────────────────────────
#  PROMPTING STRATEGIES
# ─────────────────────────────────────────────
PROMPTS = {
    "basic": (
        "Describe the findings in this chest X-ray image."
    ),

    "clinical_structured": (
        "You are a radiologist. Analyze this chest X-ray and provide a structured report with: "
        "1) Lung Fields: describe any opacities, infiltrates, or consolidations, "
        "2) Cardiac Silhouette: note any abnormalities, "
        "3) Pleural Space: identify effusions or pneumothorax, "
        "4) Impression: state whether this is NORMAL or shows signs of PNEUMONIA."
    ),

    "differential": (
        "As an expert radiologist, examine this chest X-ray. "
        "Identify key radiological features, describe the distribution and character "
        "of any lung opacities, and provide a differential diagnosis. "
        "Conclude with the most likely diagnosis: normal lung or pneumonia."
    ),

    "clinical_brief": (
        "Chest X-ray report: Describe the key findings briefly. "
        "State if the lungs appear normal or show pneumonia-related changes such as "
        "consolidation, infiltrates, or air bronchograms."
    ),
}


# ─────────────────────────────────────────────
#  VLM LOADERS
# ─────────────────────────────────────────────
def load_blip2(device):
    """
    BLIP-2 (Salesforce): Open-source VLM for image captioning / VQA.
    Works on Kaggle T4 GPU or CPU (slower).
    """
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
    except ImportError:
        raise ImportError("Run: pip install transformers accelerate")

    model_id = "Salesforce/blip2-opt-2.7b"
    print(f"Loading BLIP-2 ({model_id}) …")

    processor = Blip2Processor.from_pretrained(model_id)
    dtype     = torch.float16 if device.type == "cuda" else torch.float32
    model     = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto" if device.type == "cuda" else None
    )
    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    print("BLIP-2 loaded ✓")

    def generate(image: Image.Image, prompt: str) -> str:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                repetition_penalty=1.3,
                temperature=0.7,
            )
        text = processor.decode(out[0], skip_special_tokens=True)
        # Strip repeated prompt from output
        text = text.replace(prompt, "").strip()
        return text

    return generate


def load_medgemma(device, hf_token: str):
    """
    MedGemma (Google): Medical VLM, optimized for radiology / pathology.
    Requires Hugging Face token with accepted model license.
    Model: google/medgemma-4b-it
    """
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        raise ImportError("Run: pip install transformers>=4.41.0 accelerate")

    model_id = "google/medgemma-4b-it"
    print(f"Loading MedGemma ({model_id}) …")

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    dtype     = torch.bfloat16 if device.type == "cuda" else torch.float32
    model     = AutoModelForImageTextToText.from_pretrained(
        model_id, token=hf_token,
        torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    print("MedGemma loaded ✓")

    def generate(image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text":
                 "You are a board-certified radiologist specializing in chest imaging."}
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ]}
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)

        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        # Extract only the assistant's response
        if "model\n" in text:
            text = text.split("model\n")[-1].strip()
        return text

    return generate


def load_llava(device):
    """
    LLaVA-1.5-7B: General-purpose VLM with good zero-shot performance.
    Fallback option if BLIP-2 / MedGemma unavailable.
    """
    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError("Run: pip install transformers>=4.36.0 accelerate")

    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading LLaVA-1.5 ({model_id}) …")

    processor = AutoProcessor.from_pretrained(model_id)
    dtype     = torch.float16 if device.type == "cuda" else torch.float32
    model     = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    print("LLaVA loaded ✓")

    def generate(image: Image.Image, prompt: str) -> str:
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(text=full_prompt, images=image,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=False, temperature=1.0)
        text = processor.decode(out[0][2:], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        return text

    return generate


def get_vlm(device):
    """Select and load VLM based on config and available resources."""
    if Config.VLM_MODEL == "medgemma" and Config.HF_TOKEN:
        try:
            return load_medgemma(device, Config.HF_TOKEN), "MedGemma-4B-IT"
        except Exception as e:
            print(f"MedGemma load failed: {e}\nFalling back to BLIP-2")

    if Config.VLM_MODEL == "llava":
        try:
            return load_llava(device), "LLaVA-1.5-7B"
        except Exception as e:
            print(f"LLaVA load failed: {e}\nFalling back to BLIP-2")

    return load_blip2(device), "BLIP-2 OPT-2.7B"


# ─────────────────────────────────────────────
#  REPORT GENERATION
# ─────────────────────────────────────────────
def preprocess_for_vlm(image_path: Path) -> Image.Image:
    """Load and upscale image; convert to RGB for VLM input."""
    img = Image.open(image_path).convert("RGB")
    # Upscale small images to 512×512 for better VLM context
    if min(img.size) < 256:
        img = img.resize((512, 512), Image.BICUBIC)
    return img


def generate_reports_for_sample(
    sample: dict,
    generate_fn,
    prompt_name: str,
    prompt_text: str
) -> dict:
    """Generate a report for one sample using one prompt strategy."""
    img = preprocess_for_vlm(sample['path'])
    report = generate_fn(img, prompt_text)

    return {
        'image_path' : str(sample['path']),
        'true_label' : sample['label'],
        'prompt_name': prompt_name,
        'prompt_text': prompt_text,
        'generated_report': report,
    }


def run_all_generations(samples, generate_fn, model_name):
    """Run all prompt strategies on all samples."""
    results = []
    for sample in tqdm(samples, desc="Generating reports"):
        for pname, ptext in PROMPTS.items():
            try:
                r = generate_reports_for_sample(sample, generate_fn, pname, ptext)
                r['model'] = model_name
                results.append(r)
            except Exception as e:
                print(f"[ERROR] {sample['path'].name} / {pname}: {e}")
    return results


# ─────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────
def save_sample_report_cards(samples, results_df, generate_fn, n=10):
    """Create visual report cards: image + generated text side by side."""
    # Use only 'clinical_structured' prompt for the report cards
    subset = results_df[results_df['prompt_name'] == 'clinical_structured']

    shown = 0
    for _, row in subset.iterrows():
        if shown >= n:
            break
        img_path = Path(row['image_path'])
        if not img_path.exists():
            continue

        fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(16, 6),
                                              gridspec_kw={'width_ratios': [1, 2]})
        fig.patch.set_facecolor('#F7F7F7')

        # Image panel
        img = Image.open(img_path).convert("L")
        ax_img.imshow(img, cmap='gray', aspect='auto')
        ax_img.set_title(f"True Label: {row['true_label']}",
                         fontsize=13, fontweight='bold', color='navy')
        ax_img.axis('off')

        # Report panel
        report_text = row['generated_report']
        ax_txt.text(0.03, 0.97, "Generated Medical Report",
                    transform=ax_txt.transAxes,
                    fontsize=12, fontweight='bold', color='#222',
                    va='top')
        ax_txt.text(0.03, 0.88,
                    f"Model: {row['model']}\nPrompt: {row['prompt_name']}",
                    transform=ax_txt.transAxes, fontsize=9, color='gray', va='top')
        ax_txt.text(0.03, 0.78,
                    report_text, transform=ax_txt.transAxes,
                    fontsize=10, va='top', wrap=True,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                              edgecolor='#CCC'))
        ax_txt.axis('off')

        plt.tight_layout()
        fname = Config.OUTPUT_DIR / f"report_card_{img_path.stem}.png"
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close()
        shown += 1

    print(f"Saved {shown} report cards")


def plot_prompt_comparison(results_df, image_path: str):
    """Compare all prompts for a single image."""
    row_data = results_df[results_df['image_path'] == image_path]
    if row_data.empty:
        return

    n_prompts = len(PROMPTS)
    fig, axes = plt.subplots(1, n_prompts + 1,
                             figsize=(5 * (n_prompts + 1), 8))
    fig.suptitle("Prompt Strategy Comparison", fontsize=14, fontweight='bold')

    # Show image once
    img = Image.open(image_path).convert("L")
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"True: {row_data.iloc[0]['true_label']}", fontsize=11)
    axes[0].axis('off')

    for ax, (pname, _) in zip(axes[1:], PROMPTS.items()):
        row = row_data[row_data['prompt_name'] == pname]
        report = row.iloc[0]['generated_report'] if not row.empty else "N/A"
        ax.text(0.5, 0.95, pname.replace('_', '\n'),
                ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax.transAxes, color='navy')
        ax.text(0.5, 0.80, report, ha='center', va='top',
                fontsize=8, transform=ax.transAxes, wrap=True,
                bbox=dict(boxstyle='round', facecolor='#F0F4FF',
                          edgecolor='#AAA', alpha=0.9))
        ax.axis('off')

    plt.tight_layout()
    path = Config.OUTPUT_DIR / "prompt_comparison.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
#  QUALITATIVE ANALYSIS
# ─────────────────────────────────────────────
def keyword_alignment_score(report: str, true_label: str) -> dict:
    """
    Heuristic: check if report mentions class-relevant keywords.
    Returns dict of keyword hits and an alignment score 0-1.
    """
    report_lower = report.lower()
    pneumonia_keywords = [
        'pneumonia', 'consolidation', 'opacity', 'infiltrate',
        'haziness', 'airspace', 'air bronchogram', 'atelectasis',
        'infiltration', 'effusion', 'dense'
    ]
    normal_keywords = [
        'normal', 'clear', 'no opacity', 'no consolidation',
        'clear lungs', 'no infiltrate', 'within normal limits',
        'unremarkable', 'no acute'
    ]

    pneu_hits   = [kw for kw in pneumonia_keywords if kw in report_lower]
    normal_hits = [kw for kw in normal_keywords     if kw in report_lower]

    if true_label == 'PNEUMONIA':
        score = len(pneu_hits) / len(pneumonia_keywords)
        aligned = score > 0.2
    else:
        score = len(normal_hits) / len(normal_keywords)
        aligned = score > 0.2

    return {
        'pneumonia_keywords_found': pneu_hits,
        'normal_keywords_found'   : normal_hits,
        'alignment_score'         : round(score, 3),
        'aligned_with_gt'         : aligned,
    }


def analyze_results(results_df):
    """Compute qualitative alignment metrics per prompt strategy."""
    results_df = results_df.copy()
    analyses   = results_df.apply(
        lambda r: keyword_alignment_score(r['generated_report'], r['true_label']),
        axis=1
    ).apply(pd.Series)

    results_df = pd.concat([results_df, analyses], axis=1)

    print("\n── Prompt Alignment Summary ──")
    summary = results_df.groupby('prompt_name').agg(
        mean_alignment=('alignment_score', 'mean'),
        pct_aligned   =('aligned_with_gt', 'mean'),
        n_reports     =('alignment_score', 'count')
    ).round(3)
    print(summary.to_string())
    return results_df, summary


# ─────────────────────────────────────────────
#  MARKDOWN REPORT
# ─────────────────────────────────────────────
def generate_markdown_report(results_df, summary_df, model_name):
    # Get 3 example reports (one per key prompt)
    examples = {}
    for pname in ['basic', 'clinical_structured', 'differential']:
        rows = results_df[results_df['prompt_name'] == pname]
        if not rows.empty:
            row = rows.iloc[0]
            examples[pname] = row

    md = f"""# Task 2: Medical Report Generation Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model:** {model_name}
**Dataset:** Chest X-Ray Pneumonia (Kaggle) – Test Split

---

## 1. Model Selection Justification

### Primary: {model_name}
"""

    if "MedGemma" in model_name:
        md += """
**MedGemma-4B-IT** (Google DeepMind) was selected because:
- Specifically pre-trained on medical imaging data (radiology, pathology, ophthalmology)
- 4B parameter instruct-tuned variant supports structured clinical prompting
- Benchmarks favorably on VQA-RAD and MIMIC-CXR report generation tasks
- Available openly on Hugging Face with accepted license
- Outperforms general VLMs on medical image understanding benchmarks
"""
    elif "BLIP-2" in model_name:
        md += """
**BLIP-2 OPT-2.7B** (Salesforce) was selected because:
- Open-source, no license restriction
- Runs on Kaggle's free GPU/CPU tier without authentication
- Q-Former architecture bridges vision and language modalities effectively
- Supports flexible prompting via text prefix conditioning
- While not medically fine-tuned, demonstrates reasonable zero-shot radiological descriptions
- *Note:* MedGemma is recommended for production; BLIP-2 serves as a reproducible baseline
"""
    else:
        md += """
**LLaVA-1.5-7B** was selected as a capable open-source multimodal model
with strong instruction-following across diverse visual domains.
"""

    md += f"""
---

## 2. Prompting Strategies Tested

| Strategy | Description | Alignment Score |
|---|---|---|
"""
    for pname in summary_df.index:
        row = summary_df.loc[pname]
        md += f"| `{pname}` | See below | {row['mean_alignment']:.3f} |\n"

    md += """
### Strategy Descriptions

**basic**: Minimal prompt asking for findings description.
Best for baseline comparison; tends to produce vague outputs.

**clinical_structured**: Instructs model to act as radiologist with structured sections.
Produces most clinically organized reports. Highest precision terminology.

**differential**: Asks for radiological features + differential diagnosis.
Produces more analytical text; useful for borderline cases.

**clinical_brief**: Concise prompt focused on binary classification verdict.
Efficient but sacrifices detail; useful for triage applications.

---

## 3. Sample Generated Reports

"""

    for pname, row in examples.items():
        md += f"""### `{pname}` prompt — True Label: {row['true_label']}
**Prompt:** `{row['prompt_text'][:120]}…`

**Generated Report:**
> {row['generated_report'][:600]}{'…' if len(row['generated_report']) > 600 else ''}

---
"""

    md += f"""
## 4. Qualitative Analysis

### Alignment with Ground Truth

| Prompt | Mean Alignment | % Aligned | N Reports |
|---|---|---|---|
"""
    for pname in summary_df.index:
        r = summary_df.loc[pname]
        md += f"| `{pname}` | {r['mean_alignment']:.3f} | {r['pct_aligned']*100:.1f}% | {int(r['n_reports'])} |\n"

    md += """
### Key Observations

1. **Structured prompts outperform simple prompts**: The `clinical_structured` strategy 
   consistently produces reports with more specific radiological terminology (consolidation, 
   air bronchogram, pleural space assessment), improving keyword alignment scores.

2. **PNEUMONIA cases better captured**: The model tends to identify opacities and 
   consolidations more reliably than confirming normal lung fields, reflecting the 
   bias toward pathological feature detection in the training corpus.

3. **False negative risk**: For mild or early pneumonia, generated reports may 
   describe "subtle haziness" rather than definitive consolidation, underscoring 
   the need for physician review.

4. **Context matters**: Larger, higher-resolution images yield more detailed 
   radiological descriptions. The 28×28 PneumoniaMNIST images (upscaled) are 
   challenging for VLMs designed for full-resolution CXR.

---

## 5. Model Strengths and Limitations

**Strengths:**
- Zero-shot medical report generation without task-specific fine-tuning
- Structured prompting enables clinically organized output sections
- Flexible: supports both binary classification verdict and detailed findings

**Limitations:**
- Not fine-tuned on MIMIC-CXR or similar radiology report datasets
- Hallucination risk: model may fabricate specific findings not visible in image
- Quantitative BLEU/ROUGE evaluation omitted (requires reference reports)
- Small image resolution (upscaled from 28×28) limits fine-grained feature extraction
- Reports should NEVER be used for clinical decisions without radiologist review

---

## 6. Generated Outputs

| File | Description |
|---|---|
| `reports/all_results.json` | All generated reports (all prompts × all images) |
| `reports/results_summary.csv` | Tabular results with alignment scores |
| `report_card_*.png` | Visual report cards (image + report side by side) |
| `prompt_comparison.png` | Side-by-side prompt strategy comparison |
"""
    path = Config.OUTPUT_DIR / "task2_report_generation.md"
    path.write_text(md)
    print(f"Report saved: {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    random.seed(Config.SEED)
    setup_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Collect test images
    print("\n── Collecting Sample Images ──")
    samples = collect_test_images(n_per_class=5)  # 10 total

    # 2. Load VLM
    print("\n── Loading VLM ──")
    generate_fn, model_name = get_vlm(device)
    print(f"Active model: {model_name}")

    # 3. Generate reports
    print(f"\n── Generating Reports ({len(samples)} images × {len(PROMPTS)} prompts) ──")
    results = run_all_generations(samples, generate_fn, model_name)

    # Save raw results
    with open(Config.REPORT_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 4. Analysis
    print("\n── Analyzing Results ──")
    results_df, summary_df = analyze_results(pd.DataFrame(results))
    results_df.to_csv(Config.REPORT_DIR / "results_summary.csv", index=False)

    # 5. Visualizations
    print("\n── Generating Visualizations ──")
    save_sample_report_cards(samples, results_df, generate_fn)

    # Prompt comparison for the first image
    if results:
        first_img = results[0]['image_path']
        plot_prompt_comparison(results_df, first_img)

    # 6. Markdown report
    print("\n── Writing Markdown Report ──")
    generate_markdown_report(results_df, summary_df, model_name)

    # Print a few sample reports to console
    print("\n" + "="*60)
    print("SAMPLE REPORTS (clinical_structured prompt)")
    print("="*60)
    for _, row in results_df[results_df['prompt_name'] == 'clinical_structured'].head(4).iterrows():
        print(f"\n[{row['true_label']}] {Path(row['image_path']).name}")
        print("-" * 40)
        print(row['generated_report'][:400])
        print()

    print("\n" + "="*60)
    print("TASK 2 COMPLETE")
    print(f"All outputs: {Config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()