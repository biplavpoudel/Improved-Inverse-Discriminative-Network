# Improved Inverse Discriminative Networks

This is the non-official implement for IDN.<br><br>
The paper referenced is:

> P. Wei, H. Li and P. Hu. Inverse Discriminative Networks for Handwritten Signature Verification. [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Inverse_Discriminative_Networks_for_Handwritten_Signature_Verification_CVPR_2019_paper.pdf)



## Dataset

[CEDAR](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar): English signature dataset

[BHSig260](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E): Bengali and Hindi signature dataset

[SigComp2011](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)): ICDAR 2011 Signature Verification Competition dataset

## Instructions

Download dataset in `./dataset` using `./dataset/download_dataset.py` by specifying the url provided above.<br>
Then run `./dataset/preprocess.py` to resize and prepare pairs for training.
