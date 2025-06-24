# Inference Guide

This README provides instructions for running the inference script after completing stage two of the training process.

## Overview

The inference script processes images from a specified dataset folder and generates new images based on the trained model. This is typically done after stage two of the training pipeline has been completed.

## Usage

To run the inference process, use the following command:

Replace `FOLDER_NAME` with the same folder name used during stage two training.

```bash
# Specify the folder's name directly in the script
folder_name="0"  # Replace with your desired input folder

# Run the inference script with the specified arguments
python inference.py \
    --dataset_path datasets/$folder_name \
    --model_path outputs/$folder_name \
    --output_path generated_images/ \
    --intrinsic_anchors texture material color
```

## Notes

1. Ensure the `folder_name` variable matches the folder name used during stage two training.

2. [**Important**] The `--intrinsic_anchors` parameter must be identical to what was specified during stage two training (in this example: "texture material color").

3. Generated images will be saved to the `generated_images/` directory.

4. If the inference results after running the model aren't satisfactory, try experimenting with different values for `--reg_intrinsic_weight` (try `1`, `5e-2`, or `5e-4`, etc) or change the `--seed` value. These adjustments can often lead to better outcomes.

### Back to Main Documentation
For complete system documentation and information about all stages of the ICE pipeline, please refer to the [main README document](../README.md).