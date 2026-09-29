# !/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=1
# # ag_news
# python poc_moe.py --mode drop --max_samples 20000 --dataset_name ag_news --hot_k 4 --num_train_epochs 5 > train_drop_agnews.txt
# python poc_moe.py --mode full --max_samples 20000 --dataset_name ag_news --hot_k 4 --num_train_epochs 5 > train_full_agnews.txt
# python poc_moe.py --mode mix  --max_samples 20000 --dataset_name ag_news --hot_k 4 --num_train_epochs 5 > train_mix_agnews.txt
# # imdb
# python poc_moe.py --mode drop --max_samples 20000 --dataset_name imdb --hot_k 4 --num_train_epochs 8  > train_drop_imdb.txt
# python poc_moe.py --mode full --max_samples 20000 --dataset_name imdb --hot_k 4 --num_train_epochs 8  > train_full_imdb.txt
# python poc_moe.py --mode mix  --max_samples 20000 --dataset_name imdb --hot_k 4 --num_train_epochs 8  > train_mix_imdb.txt
# # 20news
# python poc_moe.py --mode drop --max_samples 20000 --dataset_name 20news --hot_k 4 --num_train_epochs 10 > train_drop_20news.txt
# python poc_moe.py --mode full --max_samples 20000 --dataset_name 20news --hot_k 4 --num_train_epochs 10 > train_full_20news.txt
# python poc_moe.py --mode mix  --max_samples 20000 --dataset_name 20news --hot_k 4 --num_train_epochs 10 > train_mix_20news.txt
# sst2
# python poc_moe.py --mode drop --max_samples 20000 --dataset_name sst2 --hot_k 4 --num_train_epochs 10 > train_drop_sst2.txt
# python poc_moe.py --mode full --max_samples 20000 --dataset_name sst2 --hot_k 4 --num_train_epochs 10 > train_full_sst2.txt
# python poc_moe.py --mode mix  --max_samples 20000 --dataset_name sst2 --hot_k 4 --num_train_epochs 10 > train_mix_sst2.txt
# yelp_polarity
# python poc_moe.py --mode drop --max_samples 20000 --dataset_name yelp_polarity --hot_k 4 --num_train_epochs 2 > train_drop_yelp.txt
# python poc_moe.py --mode full --max_samples 20000 --dataset_name yelp_polarity --hot_k 4 --num_train_epochs 2 > train_full_yelp.txt
# python poc_moe.py --mode mix  --max_samples 20000 --dataset_name yelp_polarity --hot_k 4 --num_train_epochs 2 > train_mix_yelp.txt
# emotion
python poc_moe.py --mode drop --max_samples 20000 --dataset_name emotion --hot_k 4 --num_train_epochs 2 > train_drop_emotion.txt
python poc_moe.py --mode full --max_samples 20000 --dataset_name emotion --hot_k 4 --num_train_epochs 2 > train_full_emotion.txt
python poc_moe.py --mode mix  --max_samples 20000 --dataset_name emotion --hot_k 4 --num_train_epochs 2 > train_mix_emotion.txt