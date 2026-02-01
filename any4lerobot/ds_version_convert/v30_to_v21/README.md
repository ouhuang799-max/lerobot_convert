# LeRobot Dataset v30 to v21

## Get started

1. Downgrade datasets:

   ```bash
   pip install "datasets<4.0.0"
   ```

   > Need to downgrade datasets first since `4.0.0` introduces `List` and `Column`.

2. Install v3.0 lerobot

   ```bash
   git clone https://github.com/huggingface/lerobot.git
   pip install -e .
   ```

2. Run the converter:
   ```bash
   python convert_dataset_v30_to_v21.py --repo-id=lerobot_dataset --root=/workspace/lerobot_dataset
   python convert_dataset_v30_to_v21.py --repo-id=UR_data_20250105_lv3.0 --root=/workspace/lerobot_data/UR_data_20250105_lv3.0
   python convert_dataset_v30_to_v21.py --repo-id=UR_20260112_Putcoinsinapiggybank_lv3.0 --root=/workspace/lerobot_data/UR_20260112_Putcoinsinapiggybank_lv3.0

   0116
   python convert_dataset_v30_to_v21.py --repo-id=UR001_20260116_Rubberstorage_lv30 --root=/workspace/lerobot_data/UR001_20260116_Rubberstorage_lv30

   0122
   python3 convert_dataset_v30_to_v21.py --repo-id=franka_data_20250105_lv30 --root=/workspace/lerobot_data/franka_data_20250105_lv30

   0125
   python3 convert_dataset_v30_to_v21.py --repo-id=UR001_260123_Drug_sorting_lv30 --root=/workspace/UR001_260123_Drug_sorting_lv30


   0131
   python3 convert_dataset_v30_to_v21.py --repo-id=ur_mixed_sorting_20260130_v30 --root=/workspace/ur_mixed_sorting_20260130_v30

   python3 convert_dataset_v30_to_v21.py --repo-id=franka_20260129_Catch_duck_lv30 --root=/workspace/franka_20260129_Catch_duck_lv30


   ```
pip install "datasets==4.1.1"