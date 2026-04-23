# U-Net Ablation

Understanding U-Net Through Ablation for Semantic Segmentation

## Introduction

Across the world, researchers are working to advance the field of computer vision. Over time, research has shifted from traditional vision methods toward approaches built on artificial intelligence and machine learning. Semantic segmentation is a fundamental task in computer vision because it enables models to understand images at the pixel level. Unlike image classification, which predicts a label for an entire image, semantic segmentation assigns a class label to each pixel. This fine-grained scene understanding is important for applications such as biomedicine, where accuracy and detailed semantic understanding are especially important (Long et al., 2015; Ronneberger et al., 2015).

Among the architectures used for semantic segmentation, U-Net has become especially influential because of its robustness in low-data environments and its ability to support strong localization and segmentation. U-Net combines an encoder, which transforms an image into a more compact representation, with a decoder, which reconstructs the original image as accurately as possible from that smaller representation. Throughout the architecture, skip connections pass information between corresponding layers to help preserve important spatial detail. Because of its strong performance with limited training data, U-Net was originally introduced in a biomedical imaging context where precise segmentation was necessary and annotated data was scarce (Ronneberger et al., 2015). Since then, U-Net has become widely used beyond that original setting. Although U-Net is now well established as a strong baseline, it remains less clear which specific components contribute most to its performance.

Because it is not always obvious which architectural components are most responsible for that performance, there has been continued interest in the relative importance of skip connections, network depth, and the decoder. Prior work often evaluates complete architectures rather than isolating the contribution of individual components. This creates a useful technical question: Are skip connections essential to U-Net’s segmentation performance? Understanding the contribution of individual architectural components is valuable because it can give researchers a clearer understanding of why these models work and how their design choices affect results. Related work on fully convolutional networks also emphasized the value of skip-style architectures for combining coarse semantic information with finer appearance information, making this question especially relevant in segmentation research (Long et al., 2015).

Rather than proposing an entirely new segmentation model, this project investigates the individual components of U-Net by analyzing how skip connections contribute to performance. The novelty of this work lies in its focused ablation of skip connections to determine how strongly they affect segmentation quality. This study helps clarify whether skip connections are truly necessary or simply conventional. The methods include training and evaluating a baseline U-Net and simplified variants that remove some or all skip connections. The dataset used is ADE20K, a large-scale benchmark for scene parsing and semantic segmentation with dense annotations across a wide range of scenes and object categories (Zhou et al., 2017). To evaluate model performance, this project uses Intersection over Union (IoU) and pixel accuracy, both of which are standard segmentation evaluation measures (Everingham et al., 2010).

To summarize the expected results, the full U-Net is expected to outperform the ablated variants because skip connections provide an additional pathway for preserving spatial information throughout the model. In particular, removing skip connections will likely reduce reconstruction quality, causing less precise boundaries and lower segmentation accuracy, which should in turn decrease IoU and pixel accuracy. These results would provide insight into how researchers can design models to leverage the advantages of skip connections, especially for reconstruction-heavy tasks where retaining fine-grained information matters. Overall, this project analyzes how U-Net performs with varying amounts of skip connections in order to better establish their contribution and highlight their advantages.

## Project Goal

This repository is a research starter scaffold for a skip-connection ablation study on ADE20K. It is not a finished benchmark reproduction yet. The current codebase is designed to make the experiment structure explicit and repeatable:

- one configurable 4-stage U-Net implementation
- one common training and evaluation pipeline
- three controlled skip-connection variants
- one README that ties the code structure to the paper-style research question

## Intended Ablation Plan

The first pass of the study stays narrow on purpose: skip connections are the only architectural variable being changed. Encoder depth, decoder depth, bottleneck width, optimization settings, image size, and evaluation metrics stay fixed across variants.

| Variant | Skip Mask (shallow to deep) | What Changes | What Stays Controlled | Why It Matters |
| --- | --- | --- | --- | --- |
| `full_skip` | `[True, True, True, True]` | No skips removed | Full encoder/decoder depth, same training loop, same loss, same preprocessing | Baseline U-Net reference |
| `highres_skip_only` | `[True, True, False, False]` | Keeps only the two highest-resolution skips | Same decoder widths and stage count; disabled skips are replaced with zeros so channel sizes stay constant | Tests whether shallow spatial detail alone explains most of the benefit |
| `no_skip` | `[False, False, False, False]` | Removes all skip information | Same encoder, bottleneck, decoder, and optimizer setup | Tests whether the decoder can recover adequate segmentation quality without encoder feature reuse |

`highres_skip_only` is the middle condition because it isolates whether the shallow, high-resolution connections carry most of the segmentation benefit while still remaining easy to interpret in the writeup.

## Experimental Protocol

Each experiment is intended to follow the same protocol:

- same ADE20K train/validation split
- same resize target (`256x256` by default in this starter)
- same optimizer (`AdamW`) and cross-entropy loss
- same epoch count, early stopping rule, and seed list (`0, 1, 2`)
- same evaluation metrics: mean IoU and pixel accuracy
- same qualitative review process: save validation mask examples for visual comparison

If your ADE20K export uses a different label convention, update `num_classes` and `label_offset` in the config before training. The starter defaults are meant to be reasonable scaffolding values, not final paper-locked settings.

## Repository Map

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | Declares the project, package layout, and Python dependencies for a plain PyTorch workflow. |
| `configs/base.yaml` | Shared defaults for data paths, image size, model widths, optimizer settings, and evaluation output locations. |
| `configs/experiments/full_skip.yaml` | Baseline U-Net experiment with all skip connections enabled. |
| `configs/experiments/highres_skip_only.yaml` | Partial ablation that keeps only the two highest-resolution skip connections. |
| `configs/experiments/no_skip.yaml` | Full skip-removal ablation. |
| `scripts/prepare_ade20k.py` | Validates an ADE20K-style folder structure and writes `train.jsonl` and `val.jsonl` metadata files. |
| `scripts/train.py` | Loads one YAML experiment config and trains that model variant. |
| `scripts/evaluate.py` | Loads a trained checkpoint, computes validation metrics, and exports qualitative prediction samples. |
| `scripts/run_ablations.py` | Runs the three planned variants across seeds `0, 1, 2` and writes summary reports to `reports/`. |
| `src/unet_ablation/data/` | ADE20K dataset discovery, metadata loading, transforms, ignore-index handling, and dataloader construction. |
| `src/unet_ablation/models/` | The configurable 4-stage U-Net and its skip-mask-driven ablation behavior. |
| `src/unet_ablation/engine/` | Training loop, validation loop, checkpointing, early stopping, and qualitative export helpers. |
| `src/unet_ablation/metrics/` | Mean IoU, pixel accuracy, and confusion-matrix helpers. |
| `src/unet_ablation/utils/` | Config loading, JSON/JSONL writing, random seeding, device selection, and image/mask export helpers. |
| `tests/` | Unit tests for metrics, model shape/skip behavior, and a smoke test for config plus validation flow. |
| `data/` | Expected local storage area for ADE20K assets and generated metadata files. |
| `artifacts/` | Generated run outputs such as checkpoints, resolved configs, and training histories. |
| `reports/` | Ablation summaries written by `scripts/run_ablations.py`. |

## Component-Level Notes

### Data

The data module expects an ADE20K-style layout and can either:

- read `metadata/train.jsonl` and `metadata/val.jsonl` if they already exist, or
- scan the raw image and annotation folders directly

Expected layout:

```text
data/ade20k/
├── images/
│   ├── training/
│   └── validation/
├── annotations/
│   ├── training/
│   └── validation/
└── metadata/
    ├── train.jsonl
    └── val.jsonl
```

The dataset loader resizes images and masks, applies a simple horizontal flip augmentation during training, converts images to tensors, and keeps ignore-index handling in one place.

### Model

The model module implements one canonical 4-stage U-Net with:

- double-convolution blocks
- max-pooling downsampling
- a bottleneck block
- transposed-convolution upsampling
- a final `1x1` segmentation head

The key ablation hook is `skip_mask`. Each boolean corresponds to one encoder stage from shallow to deep. When a skip connection is disabled, the model concatenates a zero tensor in its place so the decoder width stays fixed. That keeps the ablation focused on skip information rather than accidentally changing decoder dimensionality.

### Engine

The engine module is responsible for:

- training one experiment
- evaluating on the validation split
- saving `best.pt` and `last.pt` checkpoints
- writing per-epoch history logs
- exporting qualitative prediction examples

### Metrics

The metrics module provides:

- confusion matrix accumulation
- mean IoU
- pixel accuracy

These are the primary quantitative results for the skip-connection comparison.

## Setup

Create a virtual environment, then install the package and development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Example Workflow

1. Prepare the dataset metadata:

```bash
python3 scripts/prepare_ade20k.py --root data/ade20k
```

2. Train one variant:

```bash
python3 scripts/train.py --config configs/experiments/full_skip.yaml
```

3. Evaluate one checkpoint:

```bash
python3 scripts/evaluate.py \
  --config configs/experiments/full_skip.yaml \
  --checkpoint artifacts/checkpoints/full_skip/seed_0/best.pt
```

4. Run the full ablation matrix across seeds:

```bash
python3 scripts/run_ablations.py
```

## Expected Outputs

Running the project should populate:

- `artifacts/runs/<experiment>/seed_<n>/resolved_config.json`
- `artifacts/runs/<experiment>/seed_<n>/history.jsonl`
- `artifacts/checkpoints/<experiment>/seed_<n>/best.pt`
- `artifacts/checkpoints/<experiment>/seed_<n>/last.pt`
- `artifacts/eval/<experiment>/seed_<n>/samples/`
- `reports/ablation_results.json`
- `reports/ablation_summary.md`

These outputs support both the quantitative comparison table and the qualitative boundary/mask inspection needed for the final paper.

## Future Extensions

Once the skip-focused study is stable, natural follow-up ablations include:

- reducing or increasing encoder depth while keeping skip usage fixed
- replacing the decoder with a simpler upsampling path
- comparing transposed convolutions with interpolation-based upsampling
- testing whether the skip ablation trends remain consistent under larger image sizes or longer training

## References

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) challenge. *International Journal of Computer Vision, 88*(2), 303-338.

Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 3431-3440).

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In N. Navab, J. Hornegger, W. M. Wells, & A. F. Frangi (Eds.), *Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015* (pp. 234-241). Springer.

Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., & Torralba, A. (2017). Scene parsing through ADE20K dataset. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 5122-5130).

## Use of ChatGPT

I used ChatGPT to help generate citations in APA format and to create an initial outline for organizing my introduction. I reviewed and edited the content myself and used it only as a support tool for structure and formatting.
