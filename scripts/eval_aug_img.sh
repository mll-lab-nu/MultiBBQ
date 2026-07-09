img_aug_types=(
    "brightness_up"
    "brightness_down"
    "compression"
    "contrast_up"
    "contrast_down"
    "noise"
    "resize_l"
    "resize_s"
)

models=(
"llava-hf/llava-v1.6-mistral-7b-hf"
"Qwen/Qwen2.5-VL-3B-Instruct"
"OpenGVLab/InternVL3_5-1B"
"google/gemma-3-4b-it"
"deepseek-ai/deepseek-vl-1.3b-chat"


"adept/fuyu-8b"
"Salesforce/blip2-opt-6.7b"
"Qwen/Qwen2.5-VL-7B-Instruct"
"Qwen/Qwen2.5-VL-32B-Instruct"
"OpenGVLab/InternVL3_5-8B"
"openbmb/MiniCPM-V-4_5"
"google/gemma-3-12b-it"

)


for img_aug_type in "${img_aug_types[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING IMAGE: $img_aug_type"
    echo "----------------------------------------------------"
    for model in "${models[@]}"; do
        echo "----------------------------------------------------"
        echo "EVALUATING MODEL: $model"
        echo "----------------------------------------------------"
        multibbq run --experiment aug_img "$model" --textual_context true --ambiguous true --negative true --img_aug_type "$img_aug_type"
        multibbq run --experiment aug_img "$model" --textual_context true --ambiguous true --negative false --img_aug_type "$img_aug_type"
        multibbq run --experiment aug_img "$model" --textual_context true --ambiguous false --negative true --img_aug_type "$img_aug_type"
        multibbq run --experiment aug_img "$model" --textual_context true --ambiguous false --negative false --img_aug_type "$img_aug_type"
        multibbq run --experiment aug_img "$model" --textual_context false --ambiguous true --negative true --img_aug_type "$img_aug_type"
        multibbq run --experiment aug_img "$model" --textual_context false --ambiguous true --negative false --img_aug_type "$img_aug_type"
    done
done