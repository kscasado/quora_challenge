# Machine Learning Engineer Nanodegree
## Capstone Project
Kevin Casado
April 25th, 2017

## I. Definition


### Project Overview

Natural Language Processing(NLP) has gotten a lot of attention lately with the improvements in deep learning. The visible impacts are in products such as googles translational application or in the resurgance of chat bots. One area that has also been generating a lot of interest is through duplicate question detection. The idea is to be able to tell if two questions are asking the same thing. Current implementations of doing so primarily use extensive feature engineering and an ensemble of machine learning models. However nobody has been able to give a definitive deep learning method that solves this problem.

### Problem Statement
The problem I am going to try and solve is being able to detect whether two questions are asking the same question. Given question 1 and question 2 my machine learning model will output either a 0 (Not duplicate) or a 1 (duplicate). I will attempt to solve this using deep learning and to be more specific using variations of Recurrent Neural Networks. (https://en.wikipedia.org/wiki/Recurrent_neural_network) Quora has realeased a dataset that contains over 400 thousand questions pairs with nearly half being duplicate pairs.

### Metrics
For this project I will be using the F1 score (https://en.wikipedia.org/wiki/F1_score) to calculate the effectiveness of my model. I am choosing the F1 score because it is a well known way of incorporating not only the accuracy of the model but also the ability to have good recall and precision. I will be using the first formula without adjusting beta and simply using two as the multiplier.


## II. Analysis


### Data Exploration
The dataset that I will be using is the data provided by quora in (https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs). In total it has  400 thousand question pairs with 63% being non duplicates. When not removing any words the average length is 11 words and the maximum is 248 with a standard deviation of 5. Without removing any words the total vocabulary size is 110 thousand. For some reason when first reading in the dataset there was one row with Nan values so that was removed in the beginning. There also were some instances of brackets or braces that lead me to also remove those.


### Exploratory Visualization

### Algorithms and Techniques
For NLP recurrent neural networks have shown to be beneficial because they are able to remember previous inputs. This helps in instances where the first word of a sentence plays a role on what is being said in the last word. A simple example can be seen where you say "I am not happy". In this instance if you are using a simple bag of words method you would not be able to grasp the fact that not came before happy and thus negated the meaning. When it comes to Recurrent neural networks there are two basic types of cells, Long Short Term Memory (LSTM) or Gated Recurrent Unit (GRU) I will try using both but expect LSTM's to generate better results as they have shown to typically give better results. I will also be using an embedding layer containing containing pre-trained word2vec embeddings. As well as fully connected layers at the end before a sigmoid activation. Since most feature engineering methods use things like euclidean distance or cosine distance I will incorporate this into the representation given after the RNN. I will also try using attention to see if this generates better results as it has shown to help in many other situations.
### Benchmark
Since the dataset has more items that are not duplicate the baseline would be anything above 63% accuracy. This is because when looking at the value counts of is_duplicate 63% are zero. However the values I will be shooting for are the ones that quora has published recieving in there own internal work. https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning The best f1 score that they get is 88%.

## III. Methodology


### Data Preprocessing
In NLP there are many wayts to preprocess data so I will discuss some of the approaches I will try. Stemming and Lemmatization (https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) is a way to reduce the total vocabulary of the corpus that you are working with. One of the issues when dealing with NLP is how sparse the data is. In a image you can think of the possible matrix as a L x W matrix where the image has a resolution of L x W. Each value within that matrix can then be typically from 0 - 255 if you are dealing with the typical RGB values. However in NLP there are a few things to consider, The obvious consideration is whether you are looking at words and what you are denoting to be a sequence length. This can translate easily to the example of an image. However in the example of NLP it is not a simple 0-255, since words are so sparse you then are looking at a possible V features where V is the amount of unique words in the corpus. Typically this is in the hundreds of thousands or millions.

Another way of decreasing the vocabulary of the corpus is to remove all of the stop words (http://searchmicroservices.techtarget.com/definition/stop-word) these are very common words that have shown to have very little impact on the meaning of a sentence. After doing some analysis on the length of the sentence statements I was able to see that using sentence lengths of 30 encompasses a large majority of the words used in each sentence.

Word2Vec (https://en.wikipedia.org/wiki/Word2vec) embedding is currently the best way to try and attack the sparsity that is present in NLP. I will use the Glove pretrained word vectors (https://nlp.stanford.edu/projects/glove/) to initialize the values for each words and then I will train an embedding layer on top of this. I will explain the embedding layer in the implementation portion.

### Implementation
As discussed above I used all of the preprocessing methods before implementing the model. For the model I used keras as it allowed me to easily modify and run many instances without worrying about graph re instantiation or other various issues. I was still using the tensorflow backend and used tensorflow for some of the custom implementations. For the model the final result followed the illustration below. Some of the things I played around with were adding the cosine distance and euclidean distance between the two LSTM layers as well as using deeper representations with up to 3 LSTM layers all getting gradually smaller.

For the basic structure of the graph I used an embedding layer which both questions used and then fed this into two separate LSTM layers for each question. After the LSTM representation this was fed into a concatenated layer of the two representations and then fed into 2 fully connected layers with dropout and batch normalization before a sigmoid activation. I used batch normalization (https://arxiv.org/abs/1502.03167) because it has proven to help models converge faster and receive more accurate results. Dropout was used in order to try and prevent over fitting for the model. I tried to play with using different loss functions such as documented in this paper (http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) however in the end the loss that worked best was the binary cross entropy and the best optimization was the adam optimization.




### Refinement
There were many things that I tried to tweak in preprocessing as well as in implementing the model. Initially I removed all stop words and lemmatized all words which brought my vocabulary down to 70 thousand unique words. I also played with adding more LSTM layers as well as adding attention. None of these improved the model by any noticeable measure. They all gave me an accuracy around 82% on the validation data. With a f1 score around 75%. The accuracy score was always significantly higher because my model typically did a better job of predicting the non duplicate question since there was a class imbalance in the data.

## IV. Results


### Model Evaluation and Validation
The model I chose at the end was the most basic as the improvements made by making a deeper model took typically two times longer to compute and only improved 1-2 percent. So the final model I used used a simple embedding layer and 1 layer LSTM with attention feeding into fully connected layers.  Since the evaluation dataset I was using was still thirty thousand question pairs this shows that the model would perform well on a wide range of quora questions.

### Justification
The final results receive an accuracy score of 82% and a f1 score of 75%. This is higher than the benchmark I initially created. It does not hit the mark that quora was claiming but they do not state if their results are done on the same dataset. Perhaps with more data this model could do a better job of predicting. This solution does a good job of predicting whether questions are duplicates or not but I do not think quora would want to implement a solution that only had a 82% accuracy. It has been said by google that they do not go to production without at least 95% accuracy.


## V. Conclusion


### Free-Form Visualization

### Reflection
When working on this project there were some clear processes that took overly long. For refinement of the model this by far took the longest. As making the model deeper and deeper required longer and longer training time. Typically I would work for twenty minutes before I went to bed and had the model run over night to try and get results for the next day. This clearly illustrates why there is such a big push towards TPU's by google and trying to push the state of the art as far as hardware is concerned. I was using a GTX 1070 which is considered mid tier as far as current GPU's go and training on a GTX 1080 also did not improve training time that much. 
The part that was actually the most enjoyable was cleaning and munging the data. Typically this is the part that engineers dislike the most but since the feedback comes much faster then training a model I found it much more enjoyable. However this might change when you move more into big data with things like spark or h2o. Also using spacy was a lot easier than using nltk or gensim. In the past I have used a combination of gensim and nltk modules for python but spacy was significantly faster as well as much simpler to use.
I don't think this solution is good enough for a production environment but it does show the capability of what deep learning can do in this area. For it to be in a production environment it would need to be tweaked further and contain more feature engineering to be more accurate.
### Improvement
I think there could of been more feature engineering. The only real feature engineering done in my solution was using word vectors as opposed to raw text data and using cosine similarity and euclidean distance. Otherwise the purpose of deep learning typically is that the model will take care of the feature engineering. I think someone with domain expertise would better be able to construct features that would provide better signals for the model. I think that this solution is a simple benchmark that can easily be implemented and tested against for others to improve on.
