# Having Fun Hugging Face

[] `conda activate hugging-face` to activate the environment. Also select the correct kernel in Jupyter Notebook.

[] [Hugging Face Tutorial Link](https://huggingface.co/docs/transformers/pipeline_tutorial)

- restart the ipynb kernal after downloading new libraries and dependencies.... this has cost you time!



## Machine Learning / Deep Learning 

**Tokenization** - Before going deep into any Machine Learning or Deep Learning Natural Language Processing models, every practitioner
should find a way to map raw input strings to a representation understandable by a trainable model. This process is called tokenization.

**Preprocessing** - Preprocessing is the process of converting raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.

**Embedding** - Embedding is the process of representing a word or sentence in vector form. The main motive of representing words in vector form is to bring the words which are similar in meaning closer together in vector space.

**Modeling** - The process of training a machine learning model involves providing an ML algorithm (that is, the learning algorithm) with training data to learn from. The term ML model refers to the model artifact that is created by the training process.

**Evaluation** - The evaluation of the model is the process of evaluating the performance of the model. It is the most important step in the life cycle of the model. It is the step where we get to know whether the model is working in a desired manner or not.

**Deployment** - Deployment is the process of making the model available in production. It is the final step in the life cycle of the model. It is the step where we make the model available to the end-user and get feedback from them.

**Fine-tuning** - Fine-tuning is the process of training a pre-trained model on a new dataset. Fine-tuning is widely used approach to transfer the knowledge from one domain to another domain. It is also used to improve the performance of the model on a specific task.

**Transfer Learning** - Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task. In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task.

**Hyperparameter Tuning** - Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance. Hyperparameters are the variables that control the training process and the topology of an ML model. They are typically fixed before the actual training process begins.

**Model Serving** - Model serving is the process of making a trained ML model available for inference requests, typically through a REST API endpoint. It is the last step in the ML model development workflow.

## Sources of Free Large Datasets

There are many great sources of free large datasets for learning data science, machine learning, and other AI-derived applications. Here are a few of the best:

* **Data.gov** is a repository of open data from the US government. It has a wide variety of datasets, including economic data, education data, health data, and environmental data.
[Image of Data.gov website]
* **Kaggle** is a website where data scientists can find and share datasets, collaborate on projects, and compete in challenges. It has a large collection of datasets, including many that are specifically designed for machine learning and AI applications.
[Image of Kaggle website]
* **Google Public Datasets** is a collection of open datasets that are hosted on Google Cloud Platform. It includes a variety of datasets, including image datasets, text datasets, and time series datasets.
[Image of Google Public Datasets website]
* **UCI Machine Learning Repository** is a collection of machine learning datasets that are maintained by the University of California, Irvine. It includes a variety of datasets, including image datasets, text datasets, and tabular datasets.
[Image of UCI Machine Learning Repository website]
* **OpenML** is a platform for sharing and exploring machine learning datasets. It has a large collection of datasets, including many that are specifically designed for machine learning research.
[Image of OpenML website]

These are just a few of the many great sources of free large datasets. When choosing a dataset, it is important to consider the specific topic you are interested in, the size of the dataset, and the format of the dataset.

Here are some additional tips for finding free large datasets:

* Use a search engine to search for "free large datasets" or "open data".
* Look for datasets that are hosted on reputable websites, such as those listed above.
* Read the documentation for the dataset carefully to make sure it is suitable for your needs.
* Be sure to cite the dataset properly if you use it in your work.

## Tokenization (local copy transformers/notebooks/01-training-tokenizers.ipynb)

## Tokenization doesn't have to be slow !

### Introduction

Before going deep into any Machine Learning or Deep Learning Natural Language Processing models, every practitioner
should find a way to map raw input strings to a representation understandable by a trainable model.

One very simple approach would be to split inputs over every space and assign an identifier to each word. This approach
would look similar to the code below in python

```python
s = "very long corpus..."
words = s.split(" ")  # Split over space
vocabulary = dict(enumerate(set(words)))  # Map storing the word to it's corresponding id
```

This approach might work well if your vocabulary remains small as it would store every word (or **token**) present in your original
input. Moreover, word variations like "cat" and "cats" would not share the same identifiers even if their meaning is 
quite close.

![tokenization_simple](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/tokenization.png)

### Subtoken Tokenization

To overcome the issues described above, recent works have been done on tokenization, leveraging "subtoken" tokenization.
**Subtokens** extends the previous splitting strategy to furthermore explode a word into grammatically logicial sub-components learned
from the data.

Taking our previous example of the words __cat__ and __cats__, a sub-tokenization of the word __cats__ would be [cat, ##s]. Where the prefix _"##"_ indicates a subtoken of the initial input. 
Such training algorithms might extract sub-tokens such as _"##ing"_, _"##ed"_ over English corpus.

As you might think of, this kind of sub-tokens construction leveraging compositions of _"pieces"_ overall reduces the size
of the vocabulary you have to carry to train a Machine Learning model. On the other side, as one token might be exploded
into multiple subtokens, the input of your model might increase and become an issue on model with non-linear complexity over the input sequence's length. 
 
![subtokenization](https://nlp.fast.ai/images/multifit_vocabularies.png)
 
Among all the tokenization algorithms, we can highlight a few subtokens algorithms used in Transformers-based SoTA models : 

- [Byte Pair Encoding (BPE) - Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)
- [Word Piece - Japanese and Korean voice search (Schuster, M., and Nakajima, K., 2015)](https://research.google/pubs/pub37842/)
- [Unigram Language Model - Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, T., 2018)](https://arxiv.org/abs/1804.10959)
- [Sentence Piece - A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Taku Kudo and John Richardson, 2018)](https://arxiv.org/abs/1808.06226)

Going through all of them is out of the scope of this notebook, so we will just highlight how you can use them.

### @huggingface/tokenizers library 
Along with the transformers library, we @huggingface provide a blazing fast tokenization library
able to train, tokenize and decode dozens of Gb/s of text on a common multi-core machine.

The library is written in Rust allowing us to take full advantage of multi-core parallel computations in a native and memory-aware way, on-top of which 
we provide bindings for Python and NodeJS (more bindings may be added in the future). 

We designed the library so that it provides all the required blocks to create end-to-end tokenizers in an interchangeable way. In that sense, we provide
these various components: 

- **Normalizer**: Executes all the initial transformations over the initial input string. For example when you need to
lowercase some text, maybe strip it, or even apply one of the common unicode normalization process, you will add a Normalizer. 
- **PreTokenizer**: In charge of splitting the initial input string. That's the component that decides where and how to
pre-segment the origin string. The simplest example would be like we saw before, to simply split on spaces.
- **Model**: Handles all the sub-token discovery and generation, this part is trainable and really dependant
 of your input data.
- **Post-Processor**: Provides advanced construction features to be compatible with some of the Transformers-based SoTA
models. For instance, for BERT it would wrap the tokenized sentence around [CLS] and [SEP] tokens.
- **Decoder**: In charge of mapping back a tokenized input to the original string. The decoder is usually chosen according
to the `PreTokenizer` we used previously.
- **Trainer**: Provides training capabilities to each model.

For each of the components above we provide multiple implementations:

- **Normalizer**: Lowercase, Unicode (NFD, NFKD, NFC, NFKC), Bert, Strip, ...
- **PreTokenizer**: ByteLevel, WhitespaceSplit, CharDelimiterSplit, Metaspace, ...
- **Model**: WordLevel, BPE, WordPiece
- **Post-Processor**: BertProcessor, ...
- **Decoder**: WordLevel, BPE, WordPiece, ...

All of these building blocks can be combined to create working tokenization pipelines. 
In the next section we will go over our first pipeline.