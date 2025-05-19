content=(
    "dog" \
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

length=${#content[@]}

for ((i=0; i<$length; i++)); do
    creature="${content[$i]}"
    
    echo "Inference with creature: $creature"
    python inference_my.py \
        --content "$creature"
done
