# Improved Inverse Discriminative Network

This is the non-official implementation for Improved IDN.<br><br>
The paper referenced is:

> Nurbiya Xamxidin and et al. Multilingual Offline Signature Verification Based on Improved Inverse Discriminator Network. [Semantic Scholar](https://www.semanticscholar.org/paper/Multilingual-Offline-Signature-Verification-Based-Xamxidin-Mahpirat/8e28b02fb36ecf4f96798bd16825d8e1003dbb08)



## Dataset

[CEDAR](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar): English signature dataset

[BHSig260](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E): Bengali and Hindi signature dataset

[SigComp2011](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)): ICDAR 2011 Signature Verification Competition dataset

## Instructions

Download dataset in `./dataset` using `./dataset/download_dataset.py` by specifying the url provided above.<br>
Then run `./dataset/preprocess.py` to resize and prepare pairs for training.
