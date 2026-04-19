for seed in 1 {3..10}; do
  python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/walker-run-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="../dataset/DAVIS/JPEGImages/480p" \
    --save_path="../dataset/data/walker-run-500x-train_${seed}.hdf5" \
    --num_trajectories=500 \
    --dcs_difficulty="scale_easy_video_hard" \
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed="${seed}" \
    --cuda=False
done