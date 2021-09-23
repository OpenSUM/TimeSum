# News Timeline Summarization
Modeling Importance and Coherence for Timeline Summarization

### Updates
Available
* datasets processing
* methods & ROUGE evaluation & importance/coherence evaluation code
* preprocessing instructions for new datasets


### Datasets

* T17
* Entities


### Library installation
The `news-tls` library contains tools for loading TLS datasets and running TLS methods.
To install, run:
```
pip install -r requirements.txt
pip install -e .
```

### Loading a dataset
Check out [news_tls/explore_dataset.py](news_tls/explore_dataset.py) to see how to load the provided datasets.


### Model Training

 The file run_nni.sh is an example. Modify it according to your configuration.


### Evaluation
Check out [experiments](experiments).
Check out [experiments](evaluate-imp.py) for evaluation of importance.
Check out [experiments](evaluate-coh.py) for evaluation of coherence.



### Format & preprocess your own dataset
If you have a new dataset yourself and want to use preprocess it as the datasets above, check out the [preprocessing steps here](preprocessing).



Note: part of the code content and code framework provided are from the project DATAWISE.
1. To use the submodularity method in our paper,  change the SubmodularSummarizer code in \newstls\summarizers of the news-tls project into our functions.
2. To evaluate the coherence or importance, replace the `evaluate' code in the \experiments\evaluation in the news-tls project with cohence-eval or importance-eval in this folder.