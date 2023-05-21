import pandas as pd
import re
import requests
import json
from gensim.models import Doc2Vec
from google.cloud import storage

def get_ranks(request):
    if request.method == "GET":
        return "Please provide a POST request for ranks."
    elif request.method == "POST":
        #get data from json object in post request.
        post_data = request.get_json()
        query_string = post_data['qstring']
        #query_string = 'mens white roundneck tshirt with stripes'
        #query string pre-processing.
        query_string = query_string.lower()
        query_string = re.sub('-', '', query_string)
        query_string = re.sub(r'\b\w*\d\w*\b', '', query_string)
        query_string = re.sub(r'(\s[A-Za-z]){1}\s{1}(\w+)', r'\1\2', query_string)
        query_string_list = query_string.split()
        
        #check if the string is empty.
        if query_string_list == []:
            return "Invalid string sent! please try again."
        else:
            #access model artifacts stored in GCS bucket.
            storage_client = storage.Client.create_anonymous_client()
            bucket = storage_client.bucket('mercor-project-artifacts')
            blob_model = bucket.blob('mercor_fashion_model.model')
            blob_vocab = bucket.blob('mercor_fashion_model.model.dv.vectors.npy')
            blob_dataset = bucket.blob('mercor_final_production_dataset_v1.csv')
            blob_model.download_to_filename('mercor_fashion_model.model')
            blob_vocab.download_to_filename('mercor_fashion_model.model.dv.vectors.npy')
            blob_dataset.download_to_filename('dataset.csv')

            #Load the model and the dataset for querying.
            model = Doc2Vec.load('mercor_fashion_model.model')
            df = pd.read_csv('dataset.csv')
            vec = model.infer_vector(query_string_list)
            res = model.dv.most_similar(vec)
            inds = []
            for ind, sim in res:
                inds.append(int(ind))
            res_list = list(df[df['pid'].isin(inds)]['url'])
            final_results = []
            for ind, url in enumerate(res_list):
                final_results.append((ind+1, url))
            res_dict = dict(final_results)
            res_json = json.dumps(res_dict)
            return res_json