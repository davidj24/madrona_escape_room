#!/bin/bash

# To manually stop training, do kill PID
# To find PID, run ps aux | grep "train.py"

# --- Configuration ---
CHECKPOINT_DIR="build/checkpoints"
ACTION_DUMP_PATH="build/dumped_actions"
UPDATES_PER_INFERENCE=500
TOTAL_UPDATES=3500 # Set a very high number for a long training run

# --- Stop any previous training instances of this script ---
# FIX 1: Make pkill more robust. It now kills any train.py process
# using this script's checkpoint directory, regardless of total_updates.
pkill -f "train.py.*--ckpt-dir ${CHECKPOINT_DIR}"

echo "Starting training for ${TOTAL_UPDATES} updates in the background..."

# --- Start the main training process ---
# The 'nohup' command ensures the process keeps running even if you close the terminal.
# The '&' at the end runs the command in the background.
# We redirect all output (stdout and stderr) to a log file for later inspection.
nohup python3 scripts/train.py \
    --num-worlds 8192 \
    --num-updates ${TOTAL_UPDATES} \
    --profile-report \
    --fp16 \
    --gpu-sim \
    --ckpt-dir ${CHECKPOINT_DIR} > training.log 2>&1 &

# Store the Process ID (PID) of the training script
TRAIN_PID=$!
echo "Training started with PID: ${TRAIN_PID}"

# --- Monitoring and Inference Loop ---
echo "Starting inference monitor. Will run viewer every ${UPDATES_PER_INFERENCE} updates."
next_update_check=${UPDATES_PER_INFERENCE}

while true; do
    # FIX 2: Add an explicit check to ensure the monitor loop exits
    # after the final checkpoint has been processed.
    if [ "${next_update_check}" -gt "${TOTAL_UPDATES}" ]; then
        echo "Target updates (${TOTAL_UPDATES}) reached. Stopping monitor."
        # Make sure the background training process is truly stopped.
        kill ${TRAIN_PID} > /dev/null 2>&1
        break
    fi

    # Check if the training process is still running (in case it crashes)
    if ! ps -p ${TRAIN_PID} > /dev/null; then
        echo "Training process (PID: ${TRAIN_PID}) is no longer running. Exiting monitor."
        break
    fi

    # Construct the path for the next expected checkpoint
    CKPT_PATH="${CHECKPOINT_DIR}/${next_update_check}.pth"

    # Check if the checkpoint file exists
    if [ -f "$CKPT_PATH" ]; then
        echo "-----------------------------------------------------"
        echo "Checkpoint ${next_update_check} found! Generating actions for viewer..."
        
        # Step 1: Run inference to generate and dump the actions to a file
        python3 scripts/infer.py \
            --num-worlds 1 \
            --num-steps 1000 \
            --fp16 \
            --ckpt-path "$CKPT_PATH" \
            --action-dump-path "${ACTION_DUMP_PATH}"

        # Step 2: Run the viewer with the dumped actions
        echo "Actions generated. Starting viewer..."
        ./build/viewer 1 --cpu "${ACTION_DUMP_PATH}"

        echo "Viewer closed. Resuming monitoring."
        echo "-----------------------------------------------------"

        # Set the next checkpoint to look for
        next_update_check=$((next_update_check + UPDATES_PER_INFERENCE))
    fi

    # Wait for 10 seconds before checking again to avoid spamming the CPU
    sleep 10
done

echo "Script finished."