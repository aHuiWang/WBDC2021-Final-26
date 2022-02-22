python src/prepare/pre_data.py
python src/prepare/pre_feature.py
python src/train/train_w2v.py
python src/train/add_his_multi_task.py --seed 15 --model_name DCN --log_name seed_15 --mlp_hidden_size 128 128 128 --cross_layer_num 2 --pre_trained_feat feedid userid authorid --no_need_info description_char --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 42 --model_name DCN --log_name seed_42 --mlp_hidden_size 128 128 128 --cross_layer_num 2 --pre_trained_feat feedid userid authorid --no_need_info description_char user_his_seq --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 1996 --model_name DCN --log_name seed_1996 --mlp_hidden_size 128 128 128 --cross_layer_num 2 --pre_trained_feat feedid userid authorid bgm_song_id bgm_singer_id --no_need_info description_char --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 2021 --model_name PNN2 --log_name seed_2021 --mlp_hidden_size 128 128 128 --pre_trained_feat feedid userid authorid bgm_song_id bgm_singer_id --no_need_info description_char --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 15 --model_name DCN --log_name seed_15_ckp --pre_trained_feat feedid userid authorid --no_need_info description_char --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 42 --model_name DCN --log_name seed_42_ckp --pre_trained_feat feedid userid authorid --no_need_info description_char user_his_seq --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 1996 --model_name DCN --log_name seed_1996_ckp --pre_trained_feat feedid userid authorid bgm_song_id bgm_singer_id --no_need_info description_char --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 2021 --model_name DCN --log_name seed_2021_ckp --pre_trained_feat feedid userid authorid bgm_song_id bgm_singer_id --no_need_info description_char user_his_seq --do_eval --epochs 3
python src/train/add_his_multi_task.py --seed 2048 --model_name PNN2 --log_name seed_2048_ckp --pre_trained_feat feedid userid authorid bgm_song_id bgm_singer_id --no_need_info description_char user_his_seq --do_eval --epochs 3