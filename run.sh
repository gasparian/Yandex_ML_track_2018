#!/bin/sh
#chmod u+x job.sh
python3 blend_svd.py
python3 blend_svd_pagerank.py
python3 blend_all_tfidf_svd.py
python3 blend_all_tfidf_svd_pagerank.py