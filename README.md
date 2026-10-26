# Usage

## Files

| File | Description |
|---|---|
| `model_structure.py` | Model architecture and DFESA/SCPP modules |
| `train.py` | Model training, configuration and loss functions |
| `predict.py` | Offline segmentation, bird-dropping coverage and dust quantification |

DFESA first enhances the first encoder feature. The four feature levels are then projected to 128 channels through separate MLP layers. The projected features are resized to one-quarter of the input image dimensions and concatenated in deepest-to-shallowest order. SCPP fuses the concatenated features, followed by a per-pixel classification layer. The classification output is resized to the image dimensions during training and prediction.

## Requirements

`requirements.txt` provides recommended dependency versions. Use Python 3.8 and install a PyTorch build suitable for the CPU/CUDA environment.

```bash
python -m pip install -r requirements.txt
```

## Training data

Prepare RGB images and single-channel PNG class masks. Training uses the following labels:

- `0`: background
- `1`: PV panel
- `2`: bird droppings
- `255`: ignored

Before training, group related images by PV panel and acquisition session and assign the groups to separate subsets. Image and mask filename stems must match. The training program uses these preassigned subsets.

## Training

Review `CFG` in `train.py`, then run:

```bash
python train.py --data-root data_segformer --seeds 42 123 2024
```

The three default seeds are configurable examples. The full model uses MiT-B0, a DFESA cutoff frequency coefficient of 0.1, a fusion coefficient of 0.6, a composite loss coefficient of 0.5, a region-balancing coefficient of 0.6 and a bird-dropping class weight of 1.5. Training uses 512×512 images, a batch size of 4, a learning rate and weight decay of 0.0001, a maximum of 200 epochs, data augmentation and early stopping.

The inner contour band is 3 pixels wide and is extracted by three successive 3×3 erosions. Contour-weighted cross-entropy uses K = 2. Each image's summed loss is divided by the number of classes, followed by averaging over the minibatch.

## Offline prediction

Provide a trained model directory containing its configuration, weights and saved image processor, together with a clean reference image of the same PV panel model.

```bash
python predict.py path/to/current.jpg --model-dir path/to/final_model --reference path/to/clean.jpg --output-dir results
```

Existing class masks can also be used without loading a model:

```bash
python predict.py path/to/current.jpg --mask path/to/current_mask.png --reference path/to/clean.jpg --reference-mask path/to/clean_mask.png --output-dir results
```

Each mask must match its original image dimensions. The largest panel is selected by default. Use `--select auto_center` or `--select index --index 0` to select another target. The current and clean reference panels are normalized separately to a width of 1200 and a height of 1800 pixels.

Bird-dropping coverage is calculated for the selected panel. Bird-dropping and other invalid pixels are excluded from dust statistics. The signed floating-point CIELAB b* channels are smoothed with a 15×15 Gaussian kernel. Only positive differences from the clean reference are retained. Local responses are calculated on an 8×10 grid, and the overall DSI is weighted by valid-pixel counts. Smoothing uses valid pixels only.

Results include a class mask, JSON measurements and, when valid, rectified images and grid responses. If no panel is detected, coverage is flagged as invalid. A valid panel with no bird-dropping pixels gives zero coverage. Unreliable geometry or the absence of common valid dust pixels gives an invalid DSI. Invalid numerical values are represented by JSON `null`.
