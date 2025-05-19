content=(
    "dog2" \
    "dog5" \
    "dog6" \
    "dog7" \
    "dog8" \
    "duck_toy" \
    "grey_sloth_plushie" \
    "monster_toy" \
    "robot_toy" \
    "wolf_plushie" \
    "cat2"
)

token_name=(
    "dog" \
    "dog" \
    "dog" \
    "dog" \
    "dog" \
    "toy" \
    "plush" \
    "toy" \
    "toy" \
    "plush" \
    "cat"
)


length=${#content[@]}

for ((i=0; i<$length; i++)); do
    creature="${content[$i]}"
    token="${token_name[$i]}"
    
    echo "Training with creature: $creature and token: $token"
    python train.py \
        --instance_data_dir "dataset/$creature"  \
        --center_crop \
        --num_of_assets 1 \
        --no_prior_preservation \
        --mixed_precision bf16 \
        --use_8bit_adam \
        --initializer_tokens "$token" \
        --phase1_train_steps 400 \
        --phase2_train_steps 400 \
        --output_dir "outputs/$creature"
done

# --gradient_checkpointing \
# --enable_xformers_memory_efficient_attention \