# Specify the folder's name directly in the script
folder_name="0"  # Replace with your desired input folder

# Run the preprocess script with the specified arguments
python main_stage_two.py \
    --instance_data_dir datasets/${folder_name}/ \
    --intrinsic_anchors texture material color \
    --class_data_dir outputs/preservation_images/ \
    --step1_train_steps 800 \
    --step2_train_steps 0 \
    --output_dir outputs/${folder_name}/ \
    --use_8bit_adam \
    --set_grads_to_none \
    --noise_offset 0.1 \
    --t_dist 0.5 \
    --lambda_attention 1e-5 \
    --reg_weight 1 \
    --reg_intrinsic_weight 5e-2 \
    --pos_neg_margin 5e-2 \
    --seed 6 \
    --prior_loss_weight 1.0