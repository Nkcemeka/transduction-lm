## Audio-Conditioned Transduction Model for MIDI-Roll Denoising
This repository contains code for a system that transduces piano roll embeddings from acoustic models into MIDI event sequences through an audio-conditioned language model. The pipeline leverages audio conditioning to guide the generation of structured, interpretable MIDI events, enabling high-quality piano transcription.

## Overview

- **Input:** Audio recordings (e.g., piano performances)
- **Pipeline:**  
  1. **Acoustic Model:** Extracts note probabilities or features from audio.
  2. **Audio Conditioning:** Language model or transformer is conditioned on spectrogram/music encoder.
  3. **Output:** Sequence of MIDI events (note-on, note-off, etc.)

## Features

- **End-to-end audio-to-MIDI event transduction**
- **Audio-conditioned language model/transformer for sequence generation**
- **Flexible and modular architecture for easy experimentation**

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Additional dependencies (see `requirements.txt`)

### Installation
---
```
git clone https://github.com/Nkcemeka/transduction-lm.git
cd transduction-lm
pip install -r requirements.txt
```
---


### Training

To train the three models for onset, velocity and offset prediction, run:

---
```
./train.sh
```
---

### Inference

To run inference for the three models for onset, velocity and offset prediction, run:

---
```
./inference.sh
```
---
