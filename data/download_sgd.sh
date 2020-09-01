git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
python get_multiple_domain_dialogues.py
python delexicalize_sgd.py sgd/all
python delexicalize_sgd.py sgd/single
python delexicalize_sgd.py sgd/multiple
python preprocess_sgd.py sgd/all all
python preprocess_sgd.py sgd/single single
python preprocess_sgd.py sgd/multiple multiple

wget https://ytlin.s3-ap-northeast-1.amazonaws.com/data/sgd_mtk.zip
unzip sgd_mtk.zip
rm sgd_mtk.zip