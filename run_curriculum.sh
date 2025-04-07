# Extract all adventure_stage_*_config.json files from ./configs and sort them numerically
CONFIGS=$(find ./configs -name "adventure_stage_*_config.json" | sort -V)

for config in $CONFIGS; do
    # Extract the stage number from the config filename
    stage_num=$(echo $config | sed 's/.*adventure_stage_\([0-9]*\)_config.json/\1/')
    echo "Running stage $stage_num with config $config"

    # Run the main.py with the current config
    python main.py --use_config $config

    # Prepare for the next stage if there is one
    next_stage=$((stage_num + 1))
    next_config="./configs/adventure_stage_${next_stage}_config.json"

    if [ -f "$next_config" ]; then
        echo "Preparing for stage $next_stage"
        mkdir -p "stage_$next_stage"
        cp -r "./stage_$stage_num/Checkpoints" "./stage_$next_stage"
    fi
done