
创建的文件
1. custom_state_action_mapping_franka.py — 状态-动作映射文件
    定义如何将 HDF5 数据组合为 LeRobot 的状态和动作向量
    支持双臂 Franka（左右各 7 个关节 + 末端执行器位姿 + 夹爪）
    状态维度：28 维（7+7+6+6+1+1）

2. processors_franka.py — ROS2 消息处理器
    从 ROS2 bag 文件中提取数据
    处理 JointState（关节状态和夹爪状态）
    处理 PoseStamped（末端执行器位姿）
    自动识别左右臂（根据 topic 名称）

3. 修改了 ros2_to_lerobot_converter.py
    支持根据 topic 名称选择 processor
    向后兼容现有的 processor

pip install "datasets==4.1.1"





一步转换  ur无action


python3 ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/bag     --output-dir /workspace/lv30     --repo-id lv30     --robot-type ur_dual_arm     --task-description "Rubber storage"     --custom-processor /workspace/bag2lerbot_new/processors_ur.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30

1.20
python3 ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/UR001_20260116_Rubberstorage_bag     --output-dir /workspace/UR001_20260116_Rubberstorage_bag_new     --repo-id UR001_20260116_Rubberstorage_bag_new     --robot-type ur_dual_arm     --task-description "Rubber storage"     --custom-processor /workspace/bag2lerbot_new/processors_ur.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30

franka

python3 ros2_to_lerobot_direct.py     --bags-dir /workspace/bag     --output-dir /workspace/franka_lv30     --repo-id franka_lv30     --robot-type ur_franka_arm     --task-description "None"     --custom-processor /workspace/bag2lerbot_new/processors_franka.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_franka.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30

1.21
python3 ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/UR001_beizi     --output-dir /workspace/UR001_beizi_lv30     --repo-id UR001_20260116_Rubberstorage_bag_new     --robot-type ur_dual_arm     --task-description "Rubber storage"     --custom-processor /workspace/bag2lerbot_new/processors_ur.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30
1.22
python3 ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/UR001_bag     --output-dir /workspace/UR001_bag_lv30     --repo-id UR001_bag_lv30     --robot-type ur_dual_arm     --task-description "do something"     --custom-processor /workspace/bag2lerbot_new/processors_ur.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30

1.23
python ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/UR001_bag     --output-dir /workspace/UR001_260123_Drug_sorting_lv30     --repo-id UR001_260123_Drug_sorting_lv30     --robot-type ur_dual_arm     --task-description "Drug sorting"     --custom-processor /workspace/bag2lerbot_new/processors_ur.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30

1.24
python ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/UR001_bag     --output-dir /workspace/UR001_260123_Drug_sorting_lv30     --repo-id UR001_260123_Drug_sorting_lv30     --robot-type ur_dual_arm     --task-description "Drug sorting"     --custom-processor /workspace/bag2lerbot_new/processors_ur.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30

1.25
python ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/UR001_bag     --output-dir /workspace/UR001_260123_Drug_sorting_lv30     --repo-id UR001_260123_Drug_sorting_lv30     --robot-type ur_dual_arm     --task-description "Drug sorting"     --custom-processor /workspace/bag2lerbot_ur_noaction/processors_ur.py     --mapping-file /workspace/bag2lerbot_ur_noaction/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30





使用方法
根据 README.md，转换分为两步：
第一步：ROS2 Bag 同步

python ros2_to_lerobot_converter.py batch \
  --bags-dir=/workspace/lerobot_data/franka_data \
  --output-dir=/workspace/lerobot_output \
  --custom-processor=/workspace/bag2lerbot/processors_franka.py

python ros2_to_lerobot_converter.py batch \
  --bags-dir=/workspace/lerobot_data/franka_data_20250105 \
  --output-dir=/workspace/franka_data_20250105_synced \
  --custom-processor=/workspace/bag2lerbot/processors_franka.py




python ros2_to_lerobot_converter.py batch \
    --bags-dir=/workspace/lerobot_data/UR_data_20250105 \
    --output-dir=/workspace/lerobot_data/UR_data_20250105_hdf5 \
    --custom-processor=/workspace/bag2lerbot/processors_ur.py

0116
python ros2_to_lerobot_converter.py batch \
    --bags-dir=/workspace/lerobot_data \
    --output-dir=/workspace/lerobot_data/UR001_20260116_Rubberstorage_hdf5 \
    --custom-processor=/workspace/bag2lerbot/processors_ur.py
0119
python ros2_to_lerobot_converter.py batch \
    --bags-dir=/workspace/lerobot_data/UR001_20260116_Rubberstorage_bag \
    --output-dir=/workspace/lerobot_data/UR001_20260116_Rubberstorage_hdf5 \
    --custom-processor=/workspace/bag2lerbot/processors_ur.py

python ros2_to_lerobot_converter.py batch \
  --bags-dir=/workspace/lerobot_data/franka_data_20250105 \
  --output-dir=/workspace/lerobot_data/franka_data_20250105_synced \
  --custom-processor=/workspace/bag2lerbot/processors_franka.py

0120
python3 ros2_to_lerobot_converter.py batch \
  --bags-dir=/workspace/lerobot_data/bag \
  --output-dir=/workspace/lerobot_data/0120_lv30 \
  --custom-processor=/workspace/bag2lerbot/processors_ur.py


第二步：转换为 LeRobot 数据集


python synced_to_lerobot_converter.py \
  --input-dir /workspace/lerobot_output \
  --output-dir /workspace/lerobot_dataset \
  --repo-id franka_dual_arm \
  --fps=30 \
  --robot-type=franka_dual_arm \
  --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_franka.py \
  --use-hardware-encoding \
  --vcodec h264_nvenc \
  --crf 23 \
  --batch-size 4

python synced_to_lerobot_converter.py \
  --input-dir /workspace/franka_data_20250105_synced \
  --output-dir /workspace/franka_data_20250105_lv3.0 \
  --repo-id franka_dual_arm \
  --fps=30 \
  --robot-type=franka_dual_arm \
  --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_franka.py \
  --use-hardware-encoding \
  --vcodec h264_nvenc \
  --crf 23 \
  --batch-size 4

ur5

python synced_to_lerobot_converter.py \
    --input-dir /workspace/lerobot_data/UR_data_20250105_hdf5 \
    --output-dir /workspace/lerobot_data/UR_data_20250105_lv3.0 \
    --repo-id ur_dual_arm \
    --fps=30 \
    --robot-type=ur_dual_arm \
    --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_ur.py \
    --use-hardware-encoding \
    --vcodec h264_nvenc \
    --crf 23 \
    --batch-size 4

python synced_to_lerobot_converter.py \
    --input-dir /workspace/lerobot_data/UR_20260112_Putcoinsinapiggybank_hdf5 \
    --output-dir /workspace/lerobot_data/UR_20260112_Putcoinsinapiggybank_lv3.0 \
    --repo-id ur_dual_arm \
    --fps=30 \
    --robot-type=ur_dual_arm \
    --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_ur.py \
    --use-hardware-encoding \
    --vcodec h264_nvenc \
    --crf 23 \
    --batch-size 4

0116
python synced_to_lerobot_converter.py \
    --input-dir /workspace/lerobot_data/UR001_20260116_Rubberstorage_hdf5 \
    --output-dir /workspace/lerobot_data/UR001_20260116_Rubberstorage_lv30 \
    --repo-id ur_dual_arm \
    --fps=60 \
    --robot-type=ur_dual_arm \
    --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_ur.py \
    --use-hardware-encoding \
    --vcodec h264_nvenc \
    --crf 23 \
    --batch-size 4

0119
python synced_to_lerobot_converter.py \
    --input-dir /workspace/lerobot_data/UR001_20260116_Rubberstorage_hdf5 \
    --output-dir /workspace/lerobot_data/UR001_20260116_Rubberstorage_lv30 \
    --repo-id ur_dual_arm \
    --fps=60 \
    --robot-type=ur_dual_arm \
    --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_ur.py \
    --use-hardware-encoding \
    --vcodec h264_nvenc \
    --crf 23 \
    --batch-size 4


python synced_to_lerobot_converter.py \
  --input-dir /workspace/lerobot_data/franka_data_20250105_synced \
  --output-dir /workspace/lerobot_data/franka_data_20250105_lv30 \
  --repo-id franka_dual_arm \
  --fps=30 \
  --robot-type=franka_dual_arm \
  --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_franka.py \
  --use-hardware-encoding \
  --vcodec h264_nvenc \
  --crf 23 \
  --batch-size 4
0120
python3 synced_to_lerobot_converter.py \
    --input-dir /workspace/lerobot_data/0120_lv30 \
    --output-dir /workspace/lerobot_data/lv30_old \
    --repo-id ur_dual_arm \
    --fps=30 \
    --robot-type=ur_dual_arm \
    --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_ur.py \
    --use-hardware-encoding \
    --vcodec h264_nvenc \
    --crf 23 \
    --batch-size 4





franka test
python ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/franka_data_20250105    --output-dir /workspace/franka_data_20250105     --repo-id franka_data_20250105     --robot-type ur_dual_arm     --task-description "Catch duck"     --custom-processor /workspace/bag2lerbot_franka_noaction/processors_franka.py     --mapping-file /workspace/bag2lerbot_franka_noaction/custom_state_action_mapping_franka.py    --fps 30     --workers 2     --vcodec libsvtav1     --crf 30



python ros2_to_lerobot_converter.py batch \
  --bags-dir=/workspace/lerobot_data/franka_data_20250105 \
  --output-dir=/workspace/lerobot_data/franka_data_20250105_synced \
  --custom-processor=/workspace/bag2lerbot/processors_franka.py


python synced_to_lerobot_converter.py \
  --input-dir /workspace/lerobot_data/franka_data_20250105_synced \
  --output-dir /workspace/lerobot_data/franka_data_20250105_lv3.0 \
  --repo-id franka_dual_arm \
  --fps=30 \
  --robot-type=franka_dual_arm \
  --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_franka.py \
  --use-hardware-encoding \
  --vcodec h264_nvenc \
  --crf 23 \
  --batch-size 4

0129 有action
python ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/franka_20260129_Catch_duck    --output-dir /workspace/franka_20260129_Catch_duck_action     --repo-id franka_20260129_Catch_duck_action     --robot-type ur_dual_arm     --task-description "Catch duck"     --custom-processor /workspace/bag2lerbot_new/processors_franka.py     --mapping-file /workspace/bag2lerbot_new/custom_state_action_mapping_franka.py    --fps 30     --workers 2     --vcodec libsvtav1     --crf 30


ur


python ros2_to_lerobot_direct.py     --bags-dir /workspace/lerobot_data/ur_storge_bag_20260130     --output-dir /workspace/ur_mixed_sorting_20260130_v30     --repo-id ur_mixed_sorting_20260130_v30     --robot-type ur_dual_arm     --task-description "ur mixed sorting"     --custom-processor /workspace/bag2lerbot_new_action/processors_ur.py     --mapping-file /workspace/bag2lerbot_new_action/custom_state_action_mapping_ur.py     --fps 30     --workers 2     --vcodec libsvtav1     --crf 30

