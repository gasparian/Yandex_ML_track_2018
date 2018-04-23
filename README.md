# [Yandex_ML_track_2018](https://contest.yandex.ru/algorithm2018/contest/7914/problems/)

### Result: 29th place, mean NDCG x 100000 = 85352 

### Problem:

  Predict the most relevant reply to the context. Context usually consists of 3 replicas with several replies. Each replica has relevance and confidense - let use it's product as target variable.
  
### Solution:
  Use pretrained fastText model to represent each sequence as a concatenated embeddings of replicas and match it all with target.
  
### Pipeline:
 - Install [requirements.txt](https://github.com/gasparian/Yandex_ML_track_2018/blob/master/requirements.txt) `pip3 install -r requirements.txt`
 - Change paths in [config.py](https://github.com/gasparian/Yandex_ML_track_2018/blob/master/config.py) and then run `python3 prep.py` 
 - Download [fastText](https://github.com/facebookresearch/fastText) and build it using make.
 - Download [fastText model](https://fasttext.cc/docs/en/crawl-vectors.html) trained on wikipedia and common crawl.
 - Fill [get_vectors.txt](https://github.com/gasparian/Yandex_ML_track_2018/blob/master/get_vectors.txt) with needed paths and copy it to the fastText folder.
 - Run `bash get_vectors.txt`
 - Make numpy array from processed dataset `python3 prep_fasttext_data.py`
 - Train model `python3 fasttext_lgbm.py`
