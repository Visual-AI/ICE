# Specify the folder's name directly in the script
folder_name="0"  # Replace with your desired input folder

# Run the preprocess script with the specified arguments
python inference.py \
    --dataset_path datasets/$folder_name \
    --model_path outputs/${folder_name} \
    --output_path generated_images/ \
    --intrinsic_anchors texture material color \