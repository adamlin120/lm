wget -O DailyDialog.zip http://yanran.li/files/ijcnlp_dailydialog.zip
unzip DailyDialog.zip
rm DailyDialog.zip
mv ijcnlp_dailydialog dailydialog
cd dailydialog
unzip test.zip
unzip train.zip
unzip validation.zip
rm train.zip test.zip validation.zip