## **Stage Two**: Structured Concept Learning

### Overview
Stage Two of the ICE system focuses on structured concept learning. This stage builds upon the localized concepts identified in Stage One and learns structured representations of various intrinsic properties such as color, material, and texture.

### Usage
```bash
# Specify the folder's name directly in the script
folder_name="0"  # Replace with your desired input folder

# Run the structured concept learning script
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
    --seed 107 \
    --prior_loss_weight 1.0
```
For more detailed information on each hyperparameter, refer to `stage_two_utils/config.py`.

### Note
1. When first time running the model, it will generate the preservation images which might take around ~10 minutes. The Stage Two concept learning process takes around ~5 minutes.

2. If the inference results after running the model aren't satisfactory, try experimenting with different values for `--reg_intrinsic_weight` (try `1`, `5e-2`, or `5e-4`, etc) or change the `--seed` value. These adjustments can often lead to better outcomes.

3. To perform concept refinement by further finetuning the diffusion's UNet, set `--step2_train_steps` to `300`.

### Output
The output of Stage Two will be saved in the directory specified by `--output_dir` (in this case, `outputs/${folder_name}/`).

### Back to Main Documentation
For complete system documentation and information about all stages of the ICE pipeline, please refer to the [main README document](../README.md).