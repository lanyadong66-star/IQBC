[README.md](https://github.com/user-attachments/files/33922433/README.md)
# Model Overview

A model for document image restoration and structured text extraction. The model has been trained on a large-scale dataset of real Chinese archival documents (non-synthetic images sourced from archives) and achieves strong performance on image processing and OCR text extraction tasks involving real archival documents. The project provides directory-based batch inference, a Python API, and a local Web Demo, with support for super-resolution, perspective correction, shadow removal, background fading, text enhancement, background removal, and OCR text recognition.

## 1. Features and Processing Pipeline

```text
Input image
  → Super-resolution and detail restoration
  → Perspective correction
  → Shadow removal
  → Background fading
  → Text enhancement
  → Background removal (binarization)
  → OCR text extraction
  → Image / Markdown
```

Only enabled stages are executed. The execution order is independent of the order of keys in the configuration dictionary or the order in which options are selected in the interface.

| Feature | Configuration Parameter | Description |
| --- | --- | --- |
| Super-resolution | `use_super_resolution` | Scales both image width and height by a factor of 4 and restores image details |
| Perspective correction | `use_dewarping` | Corrects geometric distortions in documents using a document mask and the restoration network |
| Shadow removal | `use_shadow_removal` | Uses the restoration model's native shadow removal task to predict a color output image |
| Background fading | `use_background_fading` | Fades background colors based on a candidate foreground mask while preserving candidate text pixels |
| Text enhancement | `use_text_enhancement` | Darkens candidate text and foreground colors according to the specified strength |
| Background removal | `use_background_removal` | Preserves the foreground identified by the model and sets all other regions to white |
| OCR text extraction | `use_ocr` | Applies Yad_OCR to the final image and outputs Markdown |

 These features can be used independently or combined into a pipeline for archival document image restoration. When performing only shadow removal, background removal, or OCR, explicitly set `use_super_resolution=False`. When all image processing stages are disabled, OCR operates directly on the original image.

## 2. Environment and Installation

### 2.1 System Requirements

| Component | Requirement |
| --- | --- |
| Operating system | Windows / Linux; install the corresponding platform-specific PyTorch build |
| Python | Version 3.10 or later |
| Image inference | PyTorch ≥ 1.13 and the dependencies listed in `requirements.txt` |
| OCR inference | PyTorch ≥ 2.4.0, Transformers ≥ 4.57.0 and < 6, Accelerate ≥ 0.34.0 |
| OCR supporting dependencies | Pillow ≥ 10.0.0, Markdown ≥ 3.6 and < 4 |
| GPU | NVIDIA CUDA; the PyTorch build, driver, and GPU architecture must be compatible |

The complete dependency list is specified in [requirements.txt](requirements-ocr.txt). This file already includes the base image processing dependencies, so separate installation is unnecessary.

### 2.2 Installing Dependencies

Run all commands below from the project root directory, which contains `sr/`, `demo/`, and `weights/`.

Windows PowerShell:

```powershell
python -m venv .venv

# Image processing features only
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# All features, including OCR; choose this command or the one above as needed
.\.venv\Scripts\python.exe -m pip install -r requirements-ocr.txt
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Image processing features only
python -m pip install -r requirements.txt

# All features, including OCR; choose this command or the one above as needed
python -m pip install -r requirements-ocr.txt
```

If using an existing Conda or virtual environment, install the dependencies directly in that environment. Throughout the following sections, `python` refers to the interpreter with the required dependencies installed. If the Windows virtual environment is not activated, replace it with `.\.venv\Scripts\python.exe`.

## 3. Model Weights

All model weights are stored under `weights/`, with the following roles.

| Default Path | Purpose |
| --- | --- |
| `weights/SR_x4.pth` | x4 super-resolution |
| `weights/restoration.pkl` | Perspective correction, shadow removal, background fading, text enhancement, and background removal |
| `weights/document_mask.pkl` | Document region mask required for perspective correction |
| `weights/ocr/yad_OCR/` | OCR model shards, configuration, tokenizer, and image processing configuration |

Super-resolution and document mask weights are not required when performing only shadow removal, background fading, text enhancement, or background removal. These four features share the restoration model within a single image inference instance.

The OCR directory must contain the complete model package, including `config.json`, `model.safetensors.index.json`, the tokenizer, Processor, chat template, and related configuration files, as well as the following two weight shards:

```text
yad_OCR-00001-of-00002.safetensors
yad_OCR-00002-of-00002.safetensors
```

Shard names must match the mappings in the index file. The application loads models locally in offline mode and does not automatically download missing weights. After obtaining the code, prepare the required weights according to the directory structure above.

## 4. Directory-Based Batch Inference

### 4.1 Configuring Input, Output, and Feature Parameters

Batch inference is configured through the **`INFERENCE_X4_CONFIG`** dictionary at the end of [sr/x4/infer_document_x4.py](sr/x4/infer_document_x4.py).
### 4.2 Download pre-trained weights
Download the pretrained model weights from the [v1.0 release](https://github.com/lanyadong66-star/PUF-Net/releases/tag/v1.0) and place the downloaded `.pth` and `.pkl` files in the `weights/` directory.
### 4.3 Running Inference

After completing the configuration, run the following command from the project root directory:

```bash
python -B sr/x4/infer_document_x4.py
```

This starts inference.

## 5. Notes

Due to the confidentiality and security requirements of the dataset, we cannot display the image processing results. Processing performance should be assessed from the inference outputs.

## 6.Future Work

We will conduct further training to address issues such as text obscured by stains and partially occluded character strokes. Further updates will follow.

