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
For NLP recurrent neural networks have shown to be beneficial because they are able to remember previous inputs. This helps in instances where the first word of a sentence plays a role on what is being said in the last word. A simple example can be seen where you say "I am not happy". In this instance if you are using a simple bag of words method you would not be able to grasp the fact that not came before happy and thus negated the meaning. When it comes to Recurrent neural networks there are two basic types of cells, Long Short Term Memory (LSTM) or Gated Recurrent Unit (GRU) I will try using both but expect LSTM's to generate better results as they have shown to typically give better results. I will also be using an embedding layer containing containing pre-trained word2vec embeddings. As well as fully connected layers at the end before a softmax classification. Since most feature engineering methods use things like euclidean distance or cosine distance I will incorporate this into the representation given after the RNN. I will also try using attention to see if this generates better results as it has shown to help in many other situations.
### Benchmark
Since the dataset has more items that are not duplicate the baseline would be anything above 63% accuracy. This is because when looking at the value counts of is_duplicate 63% are zero. However the values I will be shooting for are the ones that quora has published recieving in there own internal work. https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning The best f1 score that they get is 88%.

## III. Methodology


### Data Preprocessing
In NLP there are many wayts to preprocess data so I will discuss some of the approaches I will try. Stemming and Lemmatization (https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) is a way to reduce the total vocabulary of the corpus that you are working with. One of the issues when dealing with NLP is how sparse the data is. In a image you can think of the possible matrix as a L x W matrix where the image has a resolution of L x W. Each value within that matrix can then be typically from 0 - 255 if you are dealing with the typical RGB values. However in NLP there are a few things to consider, The obvious consideration is whether you are looking at words and what you are denoting to be a sequence length. This can translate easily to the example of an image. However in the example of NLP it is not a simple 0-255, since words are so sparse you then are looking at a possible V features where V is the amount of unique words in the corpus. Typically this is in the hundreds of thousands or millions.

Another way of decreasing the vocabulary of the corpus is to remove all of the stop words (http://searchmicroservices.techtarget.com/definition/stop-word) these are very common words that have shown to have very little impact on the meaning of a sentence. After doing some analysis on the length of the sentence statements I was able to see that using sentence lengths of 30 encompasses a large majority of the words used in each sentence.

Word2Vec (https://en.wikipedia.org/wiki/Word2vec) embedding is currently the best way to try and attack the sparsity that is present in NLP. I will use the Glove pretrained word vectors (https://nlp.stanford.edu/projects/glove/) to initialize the values for each words and then I will train an embedding layer on top of this. I will explain the embedding layer in the implementation portion.

### Implementation
As discussed above I used all of the preprocessing methods before implementing the model. For the model I used keras as it allowed me to easily modify and run many instances without worrying about graph re instantiation or other various issues. I was still using the tensorflow backend and used tensorflow for some of the custom implementations. For the model the final result followed the illustration below. Some of the things I played around with were adding the cosine distance and euclidean distance between the two LSTM layers as well as using deeper representations with up to 3 LSTM layers all getting gradually smaller.

For the basic structure of the graph I used an embedding layer which both questions used and then fed this into two separate LSTM layers for each question. After the LSTM representation this was fed into a concatenated layer of the two representations and then fed into 2 fully connected layers with dropout and batch normalization before a softmax classification. I used batch normalization (https://arxiv.org/abs/1502.03167) because it has proven to help models converge faster and receive more accurate results. Dropout was used in order to try and prevent over fitting for the model. I tried to play with using different loss functions such as documented in this paper (http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) however in the end the loss that worked best was the binary cross entropy and the best optimization was the adam optimization.




### Refinement
There were many things that I tried to tweak in preprocessing as well as in implementing the model. Initially I removed all stop words and lemmatized all words which brough my vocabulary down to 70 thousand unique words. I also played with adding more LSTM layers as well as adding attention. None of these improved the model by any noticeable measure. They all gave me an accuracy around 82% on the validataion data.
## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_
