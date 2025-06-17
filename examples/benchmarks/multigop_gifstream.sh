# SCENE_DIR：存放场景数据的根目录。

# RESULT_DIR：存放训练输出结果的目录。

# RENDER_TRAJ_PATH：渲染轨迹（例如椭圆形路径）。

# SCENE_LIST：要处理的场景名称列表。

# ENTROPY_LAMBDA_LIST：压缩的熵率-失真权重列表（控制码率与质量的 trade-off）。

# DATA_FACTOR：采样数据比例因子。

# GOP：GOP（Group of Pictures）大小，每组有多少帧。

# TOTAL_FRAME：每个场景要处理的总帧数。

# FIRST_FRAME：起始帧编号。

# Set the directory containing the scenes
SCENE_DIR="/data1/hli/dataset/Neur3D"
# Set the directory to store results
RESULT_DIR="/data1/hli/gscodec/GIFStream_branch"
# Set the rendering trajectory path
RENDER_TRAJ_PATH="ellipse"
# List of scenes to process
SCENE_LIST="flame_salmon_1 coffee_martini sear_steak flame_steak cook_spinach cut_roasted_beef"
# List of entropy lambda values (rate-distortion tradeoff parameter)
ENTROPY_LAMBDA_LIST=(0.0005 0.001 0.002 0.004)
# Data factor for training
DATA_FACTOR=2
# Number of frames per GOP (Group of Pictures)
GOP=60
# The index of the first frame to process
FIRST_FRAME=0
# Total number of frames to process
TOTAL_FRAME=300

# 🎬 遍历每个场景（如 coffee_martini）
# Loop over each scene in the scene list
for SCENE in $SCENE_LIST;
do
    # Set TYPE based on the scene name
    # ✅ 为不同场景设置不同的模型类型
    if [ "$SCENE" = "coffee_martini" ]; then
        TYPE=neur3d_2
    elif [ "$SCENE" = "flame_salmon_1" ]; then
        TYPE=neur3d_1
    else
        TYPE=neur3d_0
    fi

    # Loop over each entropy lambda (rate)
    # 💡 遍历不同的压缩率（lambda 对应不同 rate）
    for ((RATE=0; RATE<${#ENTROPY_LAMBDA_LIST[@]}; RATE++));
    do
        # Loop over each GOP segment
        # 🎞️ 对每个 GOP 片段处理
        for ((GOP_ID=0; GOP_ID < $(((TOTAL_FRAME + GOP - 1)/GOP)) ; GOP_ID++));
            # 计算 GOP_ID 总数：(TOTAL_FRAME + GOP - 1) / GOP 即：ceil(300 / 60) = 5 个 GOP。
            # 每个 GOP 从不同帧起始：GOP_START_FRAME = 0, 60, 120, 180, 240。
        do
            echo "Running $SCENE"
            # Set experiment name and output directory
            EXP_NAME=$RESULT_DIR/${SCENE}/GOP_$GOP_ID/r$RATE
            # Calculate the starting frame for this GOP
            GOP_START_FRAME=$((FIRST_FRAME + GOP_ID * GOP ))
            # Calculate the maximum number of frames for this GOP
            MAX_GOP=$((TOTAL_FRAME - GOP_START_FRAME))
            
            # 🔹 首个 GOP（GOP_ID == 0）：从头训练
            if ((GOP_ID == 0)); then
                # If this is the first GOP, train from scratch
                # 调用：simple_trainer_GIFStream.py
                #     启用 --compression_sim 表示带有压缩仿真
                #     --rd_lambda 控制压缩质量（lambda 值）
                #     --eval_steps, --save_steps 决定评估和保存频率
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --eval_steps 7000 30000 --save_steps 7000 30000 \
                    --compression_sim --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
                    --batch_size 1 --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME

                # Run evaluation and rendering after training
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --ckpt $EXP_NAME/ckpts/ckpt_29999_rank0.pt \
                    --compression end2end  --rate $RATE \
                    --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME 
            else
                # 🔸 后续 GOP：从第 0 个 GOP 的 checkpoint 开始继续训练
                # For subsequent GOPs, continue training from first checkpoint
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --eval_steps 7000 30000 --save_steps 7000 30000 \
                    --compression_sim --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
                    --batch_size 1 --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME \
                    --ckpt $RESULT_DIR/${SCENE}/GOP_0/r$RATE/ckpts/ckpt_6999_rank0.pt --continue_training 

                # Run evaluation and rendering after training
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --ckpt $EXP_NAME/ckpts/ckpt_29999_rank0.pt \
                    --compression end2end  --rate $RATE \
                    --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME 
            fi
        done
    done
done

# Run the summary script to aggregate results
python examples/summary.py --root_dir $RESULT_DIR
