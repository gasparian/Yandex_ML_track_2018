# [Yandex_ML_track_2018](https://contest.yandex.ru/algorithm2018/contest/7914/problems/)

### Result: 29th place, mean NDCG x 100000 = 85352 

### Problem:

  Rank replies by relevance according to the context. Context usually consists of 3 replicas with several replies. Each reply has relevance and confidence - I use it's product as a target variable.
  
### Solution:
  Use pretrained fastText model to represent each sequence as concatenated replicas embeddings and match it all with target. Let's create KFold cross-validation with 10 folds, train separate simple lightgbm regressor on every iteration and then calculate mean of all models predictions to make result more "stable".
  
### Pipeline:
 - Install [requirements.txt](https://github.com/gasparian/Yandex_ML_track_2018/blob/master/requirements.txt) `pip3 install -r requirements.txt`
 - Change paths in [config.py](https://github.com/gasparian/Yandex_ML_track_2018/blob/master/config.py) and then run `python3 prep.py` 
 - Download [fastText](https://github.com/facebookresearch/fastText) and build it using make.
 - Download [fastText model](https://fasttext.cc/docs/en/crawl-vectors.html) trained on wikipedia and common crawl.
 - Fill [get_vectors.txt](https://github.com/gasparian/Yandex_ML_track_2018/blob/master/get_vectors.txt) with needed paths and copy it to the fastText folder.
 - Run `bash get_vectors.txt`
 - Make numpy array from processed dataset `python3 prep_fasttext_data.py`
 - Train model `python3 fasttext_lgbm.py`
