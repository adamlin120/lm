cd data
bash download_dailydialog.sh
bash download_multiwoz_delex.sh
bash download_personachat.sh
bash download_sgd.sh
bash download_taskmaster.sh
python preprocess_dailydialog.py
python preprocess_multiwoz.py
python preprocess_persona-chat.py
python preprocess_sgd.py sgd/delex/ all
