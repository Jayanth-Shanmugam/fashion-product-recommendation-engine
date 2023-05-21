# Mercor Vetting Project: Product Similarity Ranking
The aim of this project is to create a machine learning model that takes a product description as input and returns the top-n similar product links. The dataset for this project was gathered from public datasets available on Kaggle. For training the model, the Doc2Vec model available in the gensim library was used. And finally the model is served as a Google Cloud Function.

# Setting Up The Execution Environment
To execute the provided notebook, create a virtual Python environment using either virtualenv or conda, and install the packages in the "requirements.txt" file. Once the virtual environment has been created and the packages installed, create a sub-directory called 'datasets' to store the datasets. the datasets can be downloaded from the below links.

Myntra Dataset: https://www.kaggle.com/datasets/manishmathias/myntra-fashion-dataset
Ajio Dataset: https://www.kaggle.com/datasets/manishmathias/ajio-clothing-fashion

# Dataset Preparation
The Myntra dataset contains over 60000 products and their related information. Similarly, the Ajio dataset contains over 200000 products and their related information.
To generate the final dataset, the below pre-processing steps were applied to both datasets and finally combined to create a single dataset.
- Convert strings to lower case.
- Remove brand names from product descriptions.
- Remove punctuations.
- Attach dangling alphabets to the next word. e.g. t shirt -> tshirt, v neck -> vneck, e.t.c.
- Remove stopping words such as 'and', 'with', 'of', e.t.c.
- Remove numbers and alphanumeric strings.

# Model Training & Inferencing
The model used for training and ranking is the Doc2Vec model provided in the gensim library. The Doc2Vec model is an extension of the Word2Vec model which is used for generating a vector representation of a word that retains the words context. The Doc2Vec extends this to a sentence or a set of documents and creates a vector representation of a document or a sentence rather than a single word. To train the model, the pre-processed data is then converted to the model suitable format (A list of 'TaggedDocument' objects). 
The model is then trained for 30 epochs according to the suggestions provided in the original research paper linked below. The model is saved to a `.model` which can then be used later to get ranks. To get the ranks,the preprocessing steps are applied to the new product description, and is passed through the model to get the vector representation of it using the `model.infer_vector` function. Then to get the ranks, the `model.dv.most_similar` function is used to get the 10 most similar product descriptions.

The `model.dv.most_similar` function returns a list of tuples containing the index location and the cosine similarity score. To get the product urls, the index locations are extracted from the result and the dataset id queried to get the urls.

Le, Q. V., & Mikolov, T. (2014). Distributed Representations of Sentences and Documents. ArXiv. /abs/1405.4053 [https://arxiv.org/abs/1405.4053]

# Model Deployment
The model is deployed to Google Cloud Functions. The model artifacts such as the trained `.model` file and the dataset are stored in a Google Cloud Storage bucket. Inside the function, the model and the dataset are accessed from the bucket to serve the rankings. The model is deployed to this url https://get-ranks-mwmiuw55ra-el.a.run.app. To get the rankings, send a POST request to the url with a json object having the product description with the key 'qstring'. The function returns a json object with the top 10 similar product urls.

# References
[1] Manish Mathias. (2022). Myntra Fashion[Online]. Available:
    https://www.kaggle.com/datasets/manishmathias/myntra-fashion-dataset
    
[2] Manish Mathias. (2022). Ajio Fashion Clothing[Online]. Available:
    https://www.kaggle.com/datasets/manishmathias/ajio-clothing-fashion
    
[3] Le, Q. V., & Mikolov, T. (2014). Distributed Representations of Sentences and Documents. ArXiv. /abs/1405.4053   
