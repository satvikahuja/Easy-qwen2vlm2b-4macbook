# Qwen2-VL-2B-Instruct with Mixed Precision on Apple Silicon

This repository contains a modified version of the code from [this gist](https://gist.github.com/cavit99/811919b3e7753c925ab603b1929dbd99), enabling FP16 mixed precision inference of the Qwen2-VL-2B-Instruct model on MacBooks with Apple Silicon (M1/M2/M69 chips).

## Overview

The script allows you to process images and generate descriptions using the Qwen2-VL-2B-Instruct model with improved speed and reduced memory usage by reduced precision FP16.

## Modifications Made

The following changes have been made to the original code to enable lower precision (FP16):

- **Model Loading Adjustments:**
  - **Set** the `torch_dtype` parameter to `torch.float16` to load the model in FP16 precision.
  - **Moved** the model to the specified device using `model.to(device)`.

- **Input Casting to FP16:**
  - **Casted** input tensors to FP16 precision using `inputs.to(torch.float16)` before passing them to the model for inference.

These changes aim to reduce precision on Apple Silicon devices to enhance inference speed and reduce memory consumption.

## Requirements

- **Hardware:** MacBook with Apple Silicon (M1 or M2 chip)
- **Software:**
  - Python 3.8 or higher
  - PyTorch 2.0 or newer
  - Hugging Face Transformers library
  - Additional Python packages as per the original script (e.g., `torchvision`, `Pillow`)

## Usage

1. **Install Dependencies:**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install transformers Pillow
   ```

2. **Run the Script:**

   ```bash
   python main.py
   ```

3. **Provide Image Path:**

   - When prompted, enter the path to the image you wish to process or `'q'` to quit.

## Notes

- **Mixed Precision Limitations:**
  - The MPS backend on Apple Silicon has limited support for FP16 operations. If you encounter errors related to unsupported data types or operations, consider switching to `torch.float32` by changing the `torch_dtype` parameter and removing the input casting to FP16.

- **Performance Considerations:**
  - Mixed precision can reduce memory usage and potentially improve performance, but actual benefits may vary depending on the specific hardware and software configurations.

- **Staying Updated:**
  - Ensure that all libraries (PyTorch, Transformers) are updated to their latest versions to benefit from the most recent optimizations and support for Apple Silicon.

## Acknowledgments

- Original code by [cavit99](https://gist.github.com/cavit99/811919b3e7753c925ab603b1929dbd99)

## Disclaimer

This modified code is provided as-is. The modifications are intended to enable mixed precision inference on Apple Silicon devices. Users may need to adjust the code further based on their specific environment and requirements.
