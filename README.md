# SRNN
Author: Zeping Yu  <br />
This work is accepted by COLING 2018. <br />
[Sliced Recurrent Neural Network (SRNN)](https://arxiv.org/ftp/arxiv/papers/1807/1807.02291.pdf).  <br />
SRNN is able to get much faster speed than standard RNN by slicing the sequences into many subsequences.  <br />
The code is written in keras, using tensorflow backend. We implement the SRNN(8,2) here, and Yelp 2013 dataset is used.  <br />
keras version: 2.1.5 <br />
tensorflow version: 1.6.0 <br />
python : 2.7 <br />
If you have any question, please contact me at zepingyu@foxmail.com. <br />
<br />
The pre-trained GloVe word embeddings could be downloaded at:  <br /> 
https://nlp.stanford.edu/projects/glove/ <br />
<br />
The Yelp 2013, 2014 and 2015 datasets are at:  <br /> 
https://figshare.com/articles/Yelp_2013/6292142  <br />
https://figshare.com/articles/Untitled_Item/6292253  <br />
https://figshare.com/articles/Yelp_2015/6292334  <br />
<br />
Yelp_P, Amazon_P and Amazon_F datasets are at: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M  <br />

Here is an interesting modification of SRNN for text generation, similar to language model: https://github.com/whackashoe/srnn/tree/text-generation
