models=(
    # "openai/gpt-4o"
    # "openai/gpt-5"
    # "openai/gpt-5-mini"
    # "openai/gpt-5-nano"
    # "google/gemini-2.5-pro"
    "google/gemini-2.5-flash"
    # "google/gemini-2.5-flash-lite"

)


# gpt4o
for model in "${models[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING MODEL: $model"
    echo "----------------------------------------------------"
    multibbq run --experiment main "$model" --textual_context true --ambiguous true --negative true
    multibbq run --experiment main "$model" --textual_context true --ambiguous true --negative false
    multibbq run --experiment main "$model" --textual_context true --ambiguous false --negative true
    multibbq run --experiment main "$model" --textual_context true --ambiguous false --negative false
    multibbq run --experiment main "$model" --textual_context false --ambiguous true --negative true
    multibbq run --experiment main "$model" --textual_context false --ambiguous true --negative false
done


# imagen4ultra
for model in "${models[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING MODEL: $model"
    echo "----------------------------------------------------"
    multibbq run --experiment main "$model" --data_id "imagen4ultra_image_gen" --textual_context true --ambiguous true --negative true
    multibbq run --experiment main "$model" --data_id "imagen4ultra_image_gen" --textual_context true --ambiguous true --negative false
    multibbq run --experiment main "$model" --data_id "imagen4ultra_image_gen" --textual_context true --ambiguous false --negative true
    multibbq run --experiment main "$model" --data_id "imagen4ultra_image_gen" --textual_context true --ambiguous false --negative false
    multibbq run --experiment main "$model" --data_id "imagen4ultra_image_gen" --textual_context false --ambiguous true --negative true
    multibbq run --experiment main "$model" --data_id "imagen4ultra_image_gen" --textual_context false --ambiguous true --negative false
done