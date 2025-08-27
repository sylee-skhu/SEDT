mkdir data_uhdm
cd data_uhdm

gdown https://drive.google.com/uc?id=1bAZRuDq_bxZgxvnGtLwORV9pFKq2_kM_
tar zxvf  test.tar.gz
gdown --folder https://drive.google.com/drive/folders/1M8TKTSOo5wnJHnznu58-JBp_qnBvgzu1&confirm=t
cat train_split/train_split* | tar xvfz -