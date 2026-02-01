# LeRobot Dataset v21 to v20

## Get started

1. Install v2.1 lerobot
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    git checkout d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8
    pip install -e .
    ```

2. Run the converter:
    ```bash
    python convert_dataset_v21_to_v20.py \
        --repo-id=your_id \
        --root=your_local_dir \
        --delete-old-stats \
        --push-to-hub


        
20260201
    python convert_dataset_v21_to_v20.py \
        --repo-id=franka_20260129_Catch_duck_lv21 \
        --root=/workspace/franka_20260129_Catch_duck_lv21 \
        #   不写就不删除 --delete-old-stats \
        #   不写就是不用推 --push-to-hub

    python convert_dataset_v21_to_v20.py \
        --repo-id=ur_mixed_sorting_20260130_v20 \
        --root=/workspace/ur_mixed_sorting_20260130_v20 
    ```