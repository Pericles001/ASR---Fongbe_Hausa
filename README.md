# From Speech to Text Corpora: ASR-Based Data Acquisition for Low-Resource Fongbe and Hausa

This repository documents the end-to-end pipeline for building Automatic Speech Recognition (ASR) systems and semi-supervised text corpora for two West African languages: **Fongbe** (tonal, spoken in Benin) and **Hausa** (non-tonal, spoken across West Africa). The project spans curated dataset construction, model fine-tuning, large-scale YouTube transcription, and resource release.

> **Note:** This repository serves as supplementary documentation for our Interspeech 2026 submission. All identifying information has been anonymized for double-blind review.

---

## Table of Contents

- [Overview](#overview)
- [1. Curated Fongbe Speech Dataset](#1-curated-fongbe-speech-dataset)
  - [1.1 Motivation](#11-motivation)
  - [1.2 Data Sources](#12-data-sources)
  - [1.3 Dataset Statistics](#13-dataset-statistics)
  - [1.4 Schema](#14-schema)
  - [1.5 Dataset Construction Pipeline](#15-dataset-construction-pipeline)
  - [1.6 Key Findings During Construction](#16-key-findings-during-construction)
- [2. Fine-Tuned ASR Models](#2-fine-tuned-asr-models)
  - [2.1 Fongbe: MMS-300M Fine-Tuned](#21-fongbe-mms-300m-fine-tuned)
  - [2.2 Hausa: Whisper-Small (NCAIR)](#22-hausa-whisper-small-ncair)
- [3. YouTube Video Dataset](#3-youtube-video-dataset)
  - [3.1 Collection Methodology](#31-collection-methodology)
  - [3.2 Dataset Summary](#32-dataset-summary)
  - [3.3 Domain Breakdown](#33-domain-breakdown)
- [4. Semi-Supervised ASR Corpus](#4-semi-supervised-asr-corpus)
  - [4.1 Pipeline Overview](#41-pipeline-overview)
  - [4.2 Corpus Statistics](#42-corpus-statistics)
  - [4.3 Schema](#43-schema)
  - [4.4 Sample Data](#44-sample-data)
  - [4.5 Known Limitations](#45-known-limitations)
- [5. Reproducibility](#5-reproducibility)
  - [5.1 Requirements](#51-requirements)
  - [5.2 Inference Example](#52-inference-example)
- [6. License and Citation](#6-license-and-citation)

---

## Overview

Low-resource African languages lack the text corpora needed to train language models. While spoken content in these languages is abundant on platforms like YouTube, the absence of reliable ASR systems prevents this audio from being converted into usable text. This project addresses this gap through a three-stage pipeline:

1. **Data Curation** — Constructing a unified 12.3-hour Fongbe speech dataset from two open sources.
2. **Model Fine-Tuning** — Fine-tuning MMS-300M on the curated data, achieving state-of-the-art 9.48% WER on the ALFFA benchmark (78% relative reduction from the 44.04% baseline).
3. **Large-Scale Transcription** — From a catalog of 1,553 YouTube videos (236 hours), sampling and transcribing 424 videos (45.49 hours: 24.91h Fongbe, 20.57h Hausa), producing ~6,770 transcribed segments as a semi-supervised corpus.

---

## 1. Curated Fongbe Speech Dataset

### 1.1 Motivation

Fongbe is significantly underrepresented in existing speech technology. With fewer than 100 hours of transcribed speech available globally, consolidating the available open data into a single standardized dataset is a prerequisite for meaningful ASR development. This dataset merges the two primary open-access Fongbe speech resources into a unified, train/validation/test split with consistent formatting.

### 1.2 Data Sources

| Source | Origin | Samples | Split |
|--------|--------|---------|-------|
| **Zenodo Fongbe Speech Dataset** | Crowdsourced recordings with speaker metadata (speaker, gender, age, dialect) | 3,179 | Validation |
| **ALFFA Fongbe** (train) | Kaldi-formatted read speech corpus | 8,234 | Train |
| **ALFFA Fongbe** (test) | Kaldi-formatted read speech corpus | 2,168 | Test |

### 1.3 Dataset Statistics

| Split | Samples | Duration | Source |
|-------|---------|----------|--------|
| Train | 8,234 | 5.73 h | ALFFA |
| Validation | 3,179 | 5.11 h | Zenodo |
| Test | 2,168 | 1.45 h | ALFFA |
| **Total** | **13,581** | **12.30 h** | |

### 1.4 Schema

All splits follow a unified schema:

| Column | Type | Description |
|--------|------|-------------|
| `audio` | Audio (16 kHz) | Waveform array with sampling rate |
| `text` | string | Transcription (tone-preserved) |
| `speaker_id` | string | Speaker identifier |
| `audio_filename` | string | Original WAV filename |

All audio is resampled to 16 kHz mono. Transcriptions preserve Fongbe-specific characters (`ɖ`, `ɛ`, `ɔ`) and tonal diacritics via NFD/NFC normalization.

### 1.5 Dataset Construction Pipeline

The dataset construction notebook follows these steps:

1. **Download Zenodo data** — Fetches and unzips the Fongbe Speech Dataset archive.
2. **Load metadata** — Reads the CSV with speaker demographics and file mappings.
3. **Compare with existing HF dataset** — Loads a pre-existing Fongbe speech dataset on HuggingFace and compares transcripts against the Zenodo data.
4. **Investigate differences** — Character-level and substring analysis reveals the pre-existing HF dataset contains shorter, chunked versions of the Zenodo transcripts (1,117 out of 1,395 sentences are substrings of Zenodo sentences).
5. **Upload Zenodo split** — Converts Zenodo data to a HuggingFace Dataset with audio cast to 16 kHz and pushes as the validation split.
6. **Clone ALFFA repository** — Downloads the Kaldi-formatted Fongbe ASR data.
7. **Parse Kaldi format** — Reads `wav.scp` and `text` files, extracts speaker IDs from utterance IDs.
8. **Merge all splits** — Harmonizes schemas across sources (drops Zenodo-only columns like gender/age/dialect, casts speaker ID to string) and builds a unified `DatasetDict`.
9. **Validate no data leakage** — Confirms zero filename overlap between validation and test splits.
10. **Compute duration statistics** — Reads WAV headers to calculate per-split and total hours.

### 1.6 Key Findings During Construction

- **Zenodo vs. pre-existing HF dataset:** Zero exact transcript overlap. The pre-existing dataset uses different (shorter) chunking — 1,117 of its 1,395 unique sentences appear as substrings within the 3,048 unique Zenodo sentences.
- **Character differences:** The pre-existing HF dataset uses `ε` while Zenodo uses a broader character set including accented characters (`é`, `è`), punctuation, and digits.
- **No data leakage:** Filename intersection between validation (Zenodo) and test (ALFFA) splits is empty.

---

## 2. Fine-Tuned ASR Models

### 2.1 Fongbe: MMS-300M Fine-Tuned

We fine-tuned Meta's `facebook/mms-300m` model specifically for Fongbe. The model preserves linguistic integrity by maintaining critical tonal diacritics and unique orthographic characters (e.g., `ɖ`, `ɛ`, `ɔ`, `è`, `é`).

#### Model Description

| Property | Value |
|----------|-------|
| **Architecture** | MMS (Massive Multilingual Speech), 300M parameters |
| **Base Model** | `facebook/mms-300m` |
| **Methodology** | Fine-tuned with Connectionist Temporal Classification (CTC) loss |
| **Language** | Fongbe (`fon`) |
| **Phonetic Representation** | Tone-preserved orthography using NFD/NFC normalization |
| **Special Features** | Full support for Fon-specific characters (`ɖ`, `ɛ`, `ɔ`) and tone markers |

#### Evaluation Results

The model was evaluated on the held-out ALFFA test set (2,168 utterances, with diacritics preserved):

| Metric | Score |
|--------|-------|
| **WER (Word Error Rate)** | 9.48% |
| **CER (Character Error Rate)** | 3.96% |

#### Benchmark Comparison (with diacritics)

| Model | WER (%) | CER (%) | Year |
|-------|---------|---------|------|
| Laleye et al. (Baseline) | 44.04 | — | 2016 |
| **MMS-300M-Fongbe (Ours)** | **9.48** | **3.96** | **2026** |

This represents a **78% relative WER reduction** from the prior state-of-the-art.

#### Inference Examples

| Reference | Prediction | Result |
|-----------|------------|--------|
| `gannu elɔ kpɔ hu ɖe ɔ` | `gannu elɔ kpɔ hu ɖe ɔ` | ✅ Perfect |
| `ɖɔla tεnwe` | `ɖɔla tεnwe` | ✅ Perfect |
| `ama e gbɔ mɔ ɖo nɔ ɔ nu e wε e nɔ ɖu` | `ama e gbɔ mɔ ɖo nɔ ɔ nu ɔ e nɔ ɖu` | ⚠️ Minor function-word substitution |

#### Training Procedure

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Effective Batch Size | 64 (Batch 16 × 4 Gradient Accumulation) |
| Optimizer | AdamW (Fused) |
| Epochs | 30 |
| Precision | Mixed Precision (FP16) |
| Hardware | NVIDIA H100 GPU |

**Training Logs:**

| Training Loss | Epoch | Step | Validation Loss | WER |
|--------------|-------|------|-----------------|-----|
| 26.386 | 3.1 | 500 | 1.0171 | 0.602 |
| 12.579 | 6.21 | 1000 | 0.3366 | 0.260 |
| 1.331 | 9.32 | 1500 | 0.2312 | 0.180 |
| 0.909 | 12.42 | 2000 | 0.2031 | 0.156 |
| 0.668 | 15.53 | 2500 | 0.1752 | 0.140 |
| 0.507 | 18.64 | 3000 | 0.1747 | 0.133 |
| 0.403 | 21.74 | 3500 | 0.1583 | 0.114 |
| 0.314 | 24.85 | 4000 | 0.1618 | 0.115 |
| 0.262 | 27.95 | 4500 | 0.1656 | 0.109 |

Training converged over 30 epochs: validation WER decreased from 60.2% (epoch 3) to 10.9% (epoch 28), with training loss dropping from 26.4 to 0.26. The low CER (3.96%) confirms effective preservation of Fongbe-specific characters and tonal diacritics.

#### Intended Uses and Limitations

**Intended Uses:**
- High-accuracy transcription of Fongbe speech
- Research in low-resource and tonal language modeling
- Base model for downstream Fongbe NLP tasks
- Large-scale corpus expansion via YouTube transcription

**Limitations:**
- Performance may degrade in noisy environments or with heavy background music
- Primarily trained on read/continuous speech; may require further fine-tuning for specific dialects or extremely fast colloquial speech
- The training set (12.3h) is relatively small; leveraging larger corpora like FFSTC-2 (61h) could further improve performance

### 2.2 Hausa: Whisper-Small (NCAIR)

For Hausa, we used an existing fine-tuned Whisper-Small model developed by the National Centre for Artificial Intelligence and Robotics (NCAIR), Nigeria's Federal Ministry of Communications, Innovation, and Digital Economy. This model was applied without additional fine-tuning.

| Property | Value |
|----------|-------|
| **Architecture** | Whisper-Small |
| **Base Model** | `openai/whisper-small` |
| **Language** | Hausa (`hau`) |
| **Fine-Tuned By** | NCAIR (Nigeria) |
| **Application** | Applied directly to Hausa YouTube content without further training |

---

## 3. YouTube Video Dataset

### 3.1 Collection Methodology

We cataloged 1,553 YouTube videos through two methods:
1. **Manual playlist curation** — Identifying and curating playlists of Fongbe and Hausa content across educational, cultural, music, and news domains.
2. **Automated keyword search** — Using the YouTube Data API to search for videos in both languages across multiple content categories.

Audio was extracted using `yt-dlp`, converted to mono 16 kHz WAV format, and segmented into 20–25 second chunks to prevent memory issues during model inference.

### 3.2 Dataset Summary

**Full Catalog (identified via YouTube API):**

| Language | Videos | Hours | Channels | Domains |
|----------|--------|-------|----------|---------|
| Fongbe | 423 | 65 | 234 | Education, Music, Culture |
| Hausa | 1,130 | 171 | 527 | News, Music, Culture |
| **Total** | **1,553** | **236** | **761** | |

**Processed Subset (transcribed):**

| Language | Hours | Domains |
|----------|-------|---------|
| Fongbe | 24.91 | Education, Music, Culture |
| Hausa | 20.57 | News, Music, Culture |
| **Total (424 videos)** | **45.49** | |

### 3.3 Domain Breakdown

**Fongbe Content:**
- **Educational channels:** Language instruction content including dedicated Fongbe learning series (e.g., 29 and 35 videos from two prominent educational channels)
- **Cultural/Music content:** Traditional and religious music (e.g., one channel contributing 47 videos of traditional praise music)
- **Storytelling:** Oral history and folktales from Benin

**Hausa Content:**
- **Music:** The largest category, with one music-focused channel contributing 919 videos
- **News:** International news broadcasts (BBC Hausa, VOA Hausa)
- **Entertainment:** Cultural programming, comedy, and drama

**Dataset Format:**

The YouTube video dataset is stored as a CSV with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `title` | string | Title or name of the video |
| `video_url` | string | Unique video ID or URL |
| `category` | string | Content domain/category |
| `description` | string | Video description text |

The dataset was collected using the YouTube Data API and web scraping tools (`selenium`, `beautifulsoup4`).

---

## 4. Semi-Supervised ASR Corpus

### 4.1 Pipeline Overview

The full YouTube catalog was processed through a dual-language pipeline:

```
YouTube Catalog (1,553 videos, 236h)
    │
    └── Sampled 424 videos (45.49h)
            │
            ├── Fongbe (24.91h)
            │       │
            │       └── Fine-tuned MMS-300M → Fongbe transcriptions
            │
            └── Hausa (20.57h)
                    │
                    └── NCAIR Whisper-Small → Hausa transcriptions
            │
            └── Output: ~6,770 audio-transcription pairs (Parquet format)
```

### 4.2 Corpus Statistics

| Property | Value |
|----------|-------|
| **Total Segments** | ~6,770 |
| **Audio Format** | WAV (16 kHz, Mono) |
| **Languages** | Fongbe, Hausa |
| **Annotation Type** | Semi-supervised (machine-generated) |
| **Storage Format** | Parquet (HuggingFace Datasets compatible) |
| **Segment Duration** | 20–25 seconds |

### 4.3 Schema

| Column | Type | Description |
|--------|------|-------------|
| `language` | string | Target language (`fongbe` or `hausa`) |
| `audio_file` | string | Original segment filename for reference |
| `transcription` | string | Machine-generated text transcription |
| `confidence` | float32 | Model confidence score (0–1) |
| `audio` | audio | Decoded audio array with sampling rate (16 kHz) |

### 4.4 Sample Data

Below are example transcription outputs from the corpus, illustrating the typical output quality:

**Fongbe educational content (high confidence, ≥0.95):**
```
Language: fongbe
Confidence: 0.9853
Transcription: "un ɖo ayi kpɔn fi wε un ɖo anyi kpɔn fi wε xo un ɖo mi si fi wε..."
```

**Fongbe cultural content (moderate confidence, 0.80–0.90):**
```
Language: fongbe
Confidence: 0.8529
Transcription: "akwε bo do kɔkka beɖ nbɔn e o mε e blo bɔ e lε ye ɖo ipɔ ɔ bo..."
```

Confidence scores generally range from 0.80 to 0.99, with educational content yielding higher scores and music/cultural content producing lower scores with more potential errors.

### 4.5 Known Limitations

As a semi-supervised dataset, the corpus has several known limitations:

- **No ground-truth transcripts:** Direct WER/CER computation on the YouTube data is not possible. Quality assessment relies on automatic proxy metrics (language identification accuracy, confidence scores).
- **Transcription errors:** Machine-generated transcriptions may contain errors, especially in segments with background music, heavy dialects, or code-switching.
- **Domain quality variation:** Preliminary experiments during pipeline development suggested that educational and news content produces more coherent transcriptions than music or cultural programming. A systematic qualitative analysis of the full corpus output is ongoing.
- **Observed challenges (from preliminary experiments):**
  - Fongbe videos were frequently misidentified as French by general-purpose ASR (Whisper) due to code-switching, motivating the use of a language-specific fine-tuned model.
  - Hausa content was occasionally misclassified as Arabic, particularly on musical and religious material.
  - Audio quality varied substantially across domains and channels.
- **Bias:** The dataset is biased toward online content creators and may not represent the full diversity of spoken Fongbe and Hausa.

---

## 5. Reproducibility

### 5.1 Requirements

```
Python 3.7+
torch
transformers
datasets
pandas
numpy
yt-dlp
wave (standard library)
os, glob (standard library)
```

Optional for dataset exploration:
```
IPython (for audio playback in Colab/Jupyter)
```

### 5.2 Inference Example

```python
from transformers import pipeline

# Load the fine-tuned Fongbe ASR model
asr = pipeline("automatic-speech-recognition", model="<anonymized>/mms-300m-fongbe")

# Transcribe an audio file (must be 16kHz)
transcription = asr("path_to_audio.wav")
print(transcription["text"])
```

```python
from datasets import load_dataset

# Load the semi-supervised corpus
dataset = load_dataset("<anonymized>/fongbe-hausa-asr-dataset")

# Access the first sample
sample = dataset['train'][0]
print(f"Language: {sample['language']}")
print(f"Transcription: {sample['transcription']}")
print(f"Confidence: {sample['confidence']:.4f}")
```

---

## 6. License and Citation

### License

- **Curated Fongbe Dataset:** CC-BY-NC-4.0
- **Fine-Tuned Model:** CC-BY-NC-4.0
- **Semi-Supervised Corpus:** See dataset repository for specific terms
- **YouTube Video Metadata:** Data files © Original Authors

### Citation

If you use any of these resources in your research, please cite the following:

```bibtex
@inproceedings{anonymous2026asr,
  title={From Speech to Text Corpora: Evaluating ASR-Based Data Acquisition for Low-Resource Fongbe and Hausa},
  author={Anonymous},
  booktitle={Proc. Interspeech 2026},
  year={2026},
  note={Under review}
}
```

### Acknowledgements

- **ALFFA Project** — For the foundational Fongbe speech corpus and baseline evaluation framework.
- **Zenodo Fongbe Speech Dataset** — For crowdsourced Fongbe recordings with speaker metadata.
- **NCAIR (Nigeria)** — For the pre-trained Hausa Whisper-Small ASR model.
- **Meta AI** — For the MMS-300M base model.
- **OpenAI** — For the Whisper architecture.
- **Mozilla Common Voice** — For Hausa speech data contributions.

---

## Resources Summary

| Resource | Description | Format |
|----------|-------------|--------|
| Curated Fongbe Dataset | 13,581 utterances, 12.3h, tone-preserved | HuggingFace Dataset |
| Fine-Tuned MMS-300M | Fongbe ASR, 9.48% WER | HuggingFace Model (Safetensors) |
| YouTube Video Dataset | 1,553 videos cataloged, 424 processed (45.49h) | CSV |
| Semi-Supervised Corpus | ~6,770 transcribed segments | Parquet (HuggingFace Dataset) |

> All resource links are anonymized for double-blind review and will be made public upon acceptance.
