#!/bin/bash

# Define the base paths
image_base_path="data"
output_base_path="datasets/"

# Specify the folder name directly in the script
folder_name="0"  # Replace with your input folder

# Function to handle errors
error_handler() {
    echo "Error occurred in script at line: $1"
    exit 1
}

# Trap errors and call the error_handler function
trap 'error_handler $LINENO' ERR

# Define paths for the single image
image_path="$image_base_path/$folder_name/img.jpg"
output_path="$output_base_path"

# Run the preprocess script with the specified arguments
python main_stage_one.py \
    --image_path "$image_path" \
    --output_path "$output_path"
