python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/hopper-hop-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="../dataset/DAVIS/JPEGImages/480p" \
    --save_path="../dataset/data/hopper-hop-500x-train_2.hdf5" \
    --num_trajectories=500 \
    --dcs_difficulty="scale_easy_video_hard"\
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=2 \
    --cuda=False

python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/hopper-hop-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="../dataset/DAVIS/JPEGImages/480p" \
    --save_path="../dataset/data/hopper-hop-10x-test.hdf5" \
    --num_trajectories=10 \
    --dcs_difficulty="scale_easy_video_hard"\
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=0 \
    --cuda=True


python -m scripts.sample_labeled_data \
    --data_path="../dataset/data/hopper-hop-500x-train_merged.hdf5" \
    --save_path="../dataset/data/hopper-hop-labeled-1000xtraj125.hdf5" \
    --chunk_size=1000 \
    --num_trajectories=125


python -m train_laom_labels \
    --config_path="configs/laom-labels.yaml" \
    --lapo.data_path="../dataset/data/hopper-hop-500x-train_merged.hdf5" \
    --lapo.labeled_data_path="../dataset/data/hopper-hop-labeled-1000xtraj125.hdf5" \
    --lapo.eval_data_path="../dataset/data/hopper-hop-10x-test.hdf5" \
    --bc.data_path="../dataset/data/hopper-hop-500x-train_merged.hdf5" \
    --bc.dcs_backgrounds_path="../dataset/DAVIS/JPEGImages/480p" \
    --decoder.dcs_backgrounds_path="../dataset/DAVIS/JPEGImages/480p"