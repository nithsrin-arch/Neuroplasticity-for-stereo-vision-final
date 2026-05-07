# Neuroplasticity for Stereo Vision

A Python-based research project exploring neuroplasticity-inspired deep learning approaches for stereo vision recovery and enhancement. The project combines **diffusion-based image restoration** and **self-supervised learning** to improve binocular depth perception — drawing parallels to how the human visual cortex adapts and reorganizes itself to restore stereoscopic vision.

---

## Overview

Stereo vision (stereopsis) depends on the brain's ability to extract depth from the horizontal disparity between the two eyes' images. When this system is impaired — due to amblyopia, strabismus, or other conditions — neuroplasticity allows the visual cortex to partially reorganize and recover. This project models that recovery computationally through two complementary deep learning pipelines:

- **Diffusion-based Restoration** — uses generative diffusion models to synthesize and restore high-quality stereo image pairs from degraded inputs
- **Self-Supervised Stereo Learning** — learns disparity and depth representations from unlabeled stereo image pairs without ground-truth depth annotations

---

## Repository Structure

```
Neuroplasticity-for-stereo-vision-final/
│
├── diffusion-restoration/      # Diffusion model pipeline for stereo image restoration
│   └── ...                     # Training, inference, and model scripts
│
├── self supervised/            # Self-supervised stereo depth estimation
│   └── ...                     # Training, evaluation, and loss scripts
│
├── .gitignore
└── README.md
```

---

## Modules

### 1. `diffusion-restoration`

This module applies diffusion models to restore degraded stereo image pairs. Inspired by works like DiffStereo, it:

- Leverages score-based / denoising diffusion probabilistic models (DDPMs) to reconstruct high-frequency details in stereo images
- Handles common degradation types such as blur, noise, and low-light conditions
- Uses cross-view attention to maintain stereo geometric consistency during restoration
- Outputs high-quality left–right image pairs suitable for downstream depth estimation

### 2. `self supervised`

This module trains stereo disparity and depth estimation networks without labeled ground-truth depth data. Key features include:

- Photometric reconstruction loss between synthesized and real stereo views
- Cycle-consistent training on unpaired stereo data to handle occlusions and out-of-frame regions
- Self-supervised objectives that mirror how the brain uses experience-driven plasticity to tune disparity-sensitive neurons
- Supports joint image–depth (RGBD) recovery as well as depth-only prediction modes

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Additional dependencies (install via `requirements.txt` if provided)

```bash
pip install torch torchvision
pip install -r requirements.txt   # if available
```

### Clone the Repository

```bash
git clone https://github.com/nithsrin-arch/Neuroplasticity-for-stereo-vision-final.git
cd Neuroplasticity-for-stereo-vision-final
```

### Running Diffusion Restoration

```bash
cd diffusion-restoration
python train.py      # train the diffusion model
python inference.py  # run restoration on stereo image pairs
```

### Running Self-Supervised Stereo Training

```bash
cd "self supervised"
python train.py      # self-supervised training on stereo pairs
python evaluate.py   # evaluate disparity/depth estimation
```

> **Note:** Specific arguments and dataset paths may vary. Refer to the argument parsers or config files within each module for full usage details.

---

## Motivation: Neuroplasticity & Stereo Vision

Neuroplasticity refers to the brain's ability to reorganize neural connections in response to experience or injury. In the context of stereo vision:

- Early cortical visual areas (V1, V2) encode binocular disparity and are highly plastic during critical developmental periods
- Perceptual learning studies show that stereoacuity can be improved through training, even in adults with impaired stereo vision
- This project models that adaptive process computationally: the diffusion module "restores" degraded visual input (analogous to sensory rehabilitation), while the self-supervised module learns depth representations from experience alone (analogous to unsupervised cortical adaptation)

---

## Key Concepts

| Concept | Description |
|---|---|
| **Stereopsis** | Perception of depth from binocular disparity |
| **Neuroplasticity** | Brain's ability to reorganize and adapt neural circuits |
| **Diffusion Models** | Generative models that restore images via iterative denoising |
| **Self-Supervised Learning** | Learning from data without manual labels, using proxy tasks |
| **Photometric Loss** | Pixel-level reconstruction error used as a training signal |
| **Disparity Estimation** | Computing pixel shift between left and right stereo views |

---

## Language

- **Python** (100%)

---

## Contributing

Contributions, issues, and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is released for academic and research use. Please check with the authors for licensing terms before using in commercial applications.

---

## Author

**Nithish Sriram Srinivasan**
**Akanksha Bharadwaj**
**Krishi Thirupathi**

---

## Acknowledgements

This work draws inspiration from research on:
- Diffusion-based stereo image restoration (e.g., DiffStereo)
- Self-supervised depth reconstruction from brain activity and stereo pairs
- Neuroplasticity and perceptual learning in binocular vision
