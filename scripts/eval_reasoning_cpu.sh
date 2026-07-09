models=(
    # "openai/gpt-4o"
    "openai/gpt-5"
    # "google/gemini-2.5-flash-lite"

    # "openai/gpt-5-mini"
    # "openai/gpt-5-nano"
    # "google/gemini-2.5-flash"
    # "google/gemini-2.5-pro"

)

reasoning_types=(
    'nonreasoning_w_fairness'
    'reasoning_w_fairness'
    'reasoning'
)

for rt in "${reasoning_types[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING REASONING: $rt"
    echo "----------------------------------------------------"
    for model in "${models[@]}"; do
        echo "----------------------------------------------------"
        echo "EVALUATING MODEL(REASONING): $model"
        echo "----------------------------------------------------"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous true --negative true --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous true --negative false --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous false --negative true --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous false --negative false --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context false --ambiguous true --negative true --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context false --ambiguous true --negative false --reasoning_mode "$rt"
    done
done