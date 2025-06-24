## **Stage One**: Automatic Concept Localization 

### Overview

Stage One of the ICE system performs automatic concept localization on a single input image. This stage processes images to identify and localize key concepts within it.

### Usage

To run Stage One:

1. Navigate to the ICE directory
2. Execute the script specifying the image_index (folder name containing the target image):

```bash
# Specify the folder's name directly in the script
folder_name="0"  # Replace with your desired input folder
```

Where `$folder_name` is the name of the folder containing the img.jpg file you want to process.

This would process the image located at `$image_base_path/$folder_name/img.jpg`.

### Output
The script will generate localization data for the specified image, which will be used in subsequent stages of the ICE pipeline. The output of the retrieved texts and the respective masks will be saved in `$output_base_path`.

### Back to Main Documentation
For complete system documentation and information about all stages of the ICE pipeline, please refer to the [main README document](../README.md).

