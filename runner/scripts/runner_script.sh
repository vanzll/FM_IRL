# run giril on each task with tmux
alg_name='airl'

env_names=('ant' 'hand' 'maze' 'navigation' 'pick' 'push')
additional_params=('0.00' '1.00' '25' '0.00' '1.00' '1.00') # corresponding!

for i in "${!env_names[@]}"; do
    yml_path="/home/wanzl/project/DRAIL/configs/${env_names[i]}/${additional_params[i]}/${alg_name}.yaml"
    visible_device=$(shuf -i 0-5 -n 1)
    session_name="${alg_name}_${env_names[i]}"
    # Check if session already exists, if not, create it
    if ! tmux has-session -t "$session_name" 2>/dev/null; then
        tmux new-session -d -s "$session_name"
    fi
    # Send the command to the tmux session
    tmux send-keys -t "$session_name" "bash -c 'source ~/.bashrc && conda activate  Genbot && export CUDA_VISIBLE_DEVICES=${visible_device} && utils/wandb.sh ${yml_path}" C-m
done

tmux ls

