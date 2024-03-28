
#### Preparing Data

##### Wikipedia

1. Download Wikipedia dump [here](https://dumps.wikimedia.org/).
2. Feed the xml to `dataset/process_data.py`: 
```
python process_data.py -f <path_to_xml> -o <dst_wiki> --type wiki
```

##### BookCorpus

Since the original BookCorpus adopted by BERT is no more available, we use
an updated version of publicly available books.

1. Download books [here](https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz).
2. Create destinatio directory `<dst_books>`
3. Feed the downloaded txts to `dataset/process_data.py`: <br>
```
python process_data.py -f <path_to_book_text_files> -o <dst_books> --type bookcorpus
```

##### Sharding

1. Create a new directory to contain both data `<data_dir>`.
2. Move data into the same directory `<data_dir>`: <br>
```
mv <dst_wiki>/wiki_one_article_per_line.txt <data_dir>
mv <dst_books>/bookcorpuss_one_article_per_line.txt <data_dir>
```
    
3. Generate shards into your shard directory `<shard_dir>`:
```
python shard_data.py \
    --dir <data_dir> \
    -o <shard_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1
```

3. `cd <sharebert_dir>/dataset/`
4. Generate samples from shards, into `<hdf5_dir>`:
```
python generate_samples.py \
     --dir <shard_dir> \
     -o <hdf5_dir> \
     --dup_factor 5 \
     --seed 42 \
     --vocab_file ../bert_asset/bert-base-uncased-vocab.txt \
     --do_lower_case 1 \
     --masked_lm_prob 0.15 \
     --max_seq_length 128 \
     --model_name bert-base-uncased \
     --max_predictions_per_seq 20 \
     --n_processes 16
```
