for seed in 1 2 3 4 5 6 7 8 9 10; do
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

python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/hopper-hop-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="../dataset/DAVIS/JPEGImages/480p" \
    --save_path="../dataset/data/hopper-hop-500x-test_10traj.hdf5" \
    --num_trajectories=10 \
    --dcs_difficulty="scale_easy_video_hard" \  
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=0 \
    --cuda=False



python -m scripts.sample_labeled_data \
    --data_path="path/to/full/dataset" \
    --save_path="path/to/full/dataset-labeled-1000xtraj$num_traj.hdf5" \
    --chunk_size=1000 \
    --num_trajectories=125