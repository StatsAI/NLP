# © Stats AI LLC 2023. All Rights Reserved. 
# No part of this code may be used or reproduced without express permission.

import streamlit as st
#from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torch
#from torch.autograd import Variable
import os
import math
import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import requests
import zipfile
import json

from langchain_openai import OpenAI
from typing import List
from sentence_transformers import SentenceTransformer, util

#from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)

####################################################################################################################################################

# Download and unzip images
@st.cache_resource
def download_and_unzip(url):
	response = requests.get(url)
	with open("archive.zip", "wb") as f:
        	f.write(response.content)
	
	with zipfile.ZipFile("archive.zip", "r") as zip_ref:
        	zip_ref.extractall()

@st.cache_resource
def load_data(folder_list: list):
	image_path = []
	
	for folder in folder_list:
		for root, dirs, files in os.walk(folder):
			for file in files:
				if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
					image_path.append(os.path.join(root, file))
	return image_path

@st.cache_resource
def load_embeddings():
	url = "https://github.com/StatsAI/streamlit_image_search_db/releases/download/image_search_assets/img_dict.txt"

	# Download the file
	response = requests.get(url)

	# Load the file into a string
	file_content = response.content.decode("utf-8")

	# Create a dictionary from the string
	img_dict = json.loads(file_content)
	
	img_list = list(img_dict.keys())
	#img_emb = list(img_dict.values())

	img_values = list(img_dict.values())
	img_emb= [value[0] for value in img_values]
	img_type = [value[1] for value in img_values]
		
	return img_list, img_emb, img_type


## Load Pre-trained Assets
@st.cache_resource
def load_assets():
	# Load images from a folder
	#image_list = load_data(['animals'])

	# Load indexed images
	image_list, img_emb_loaded, img_type = load_embeddings()
	img_emb_loaded = torch.tensor(img_emb_loaded)

	return image_list, img_emb_loaded, img_type

# Set up the search engine
@st.cache_resource
def load_model():
	model = SentenceTransformer("clip-ViT-B-32")
	return model

@st.cache_resource
def create_vector_db_input(_img_emb_loaded):
	img_emb_loaded = _img_emb_loaded.tolist()
	image_names = range(0,len(image_list))

	#df = pd.DataFrame(zip(image_names, image_list, img_emb_loaded), columns = ['image_name', 'image_path','embedding'])

	#df = pd.DataFrame(zip(image_names, image_list, img_emb_loaded, img_type), columns = ['image_name', 'image_path','embedding', 'type'])

	#payloads = df[['image_name', 'image_path']].fillna("Unknown").to_dict("records")
	#payloads = df[['image_name', 'image_path', 'type']].fillna("Unknown").to_dict("records")

	#client = QdrantClient(":memory:")
	#collections = client.get_collections()
	
	#client.recreate_collection(collection_name="animals", vectors_config=rest.VectorParams(size=512, distance=rest.Distance.COSINE))

	# client.upload_collection(collection_name="animals", 
	# 			 vectors=list(map(list, df["embedding"].tolist())),
	# 			 payload=payloads,
	# 			 ids=[uuid.uuid4().hex for _ in payloads])

	client = QdrantClient(
		url = st.secrets["url"], 
		api_key = st.secrets["api_key"]
		,)

	
	return client

#@st.cache_resource
def vector_db(client, animal_embedding):

	animal_embedding = animal_embedding.tolist()
	
	results = client.search(collection_name="animals",
				query_vector=animal_embedding,
				with_payload=True,
				limit=16)

	return results
####################################################################################################################################################

url = "https://github.com/StatsAI/streamlit_image_search_db/releases/download/image_search_assets/archive.zip"
download_and_unzip(url)

image_list, img_emb_loaded, img_type = load_assets()

model = load_model()

client = create_vector_db_input(img_emb_loaded)

####################################################################################################################################################

logo = Image.open('images/picture.png')
#newsize = (95, 95)
#logo = logo.resize(newsize)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
	    margin-top: -75px;
            width: 100%;
	    #margin: 0;	         		
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image(logo)

st.markdown("""
        <style>
               .block-container {
		    padding-top: 0;
                }
        </style>
        """, unsafe_allow_html=True)

#st.write('')
#st.write('')
st.title("Reverse Image Search via OpenAI + CLIP + Qdrant Vector Database")
#st.write("This app performs reverse image search using OpenAI's CLIP + Qdrant Vector Database")

images_recs = st.sidebar.slider(label = 'Image Search: Select an animal using the slider', min_value = 1,
                          max_value = 5400,
                          value = 1859,
                          step = 1)

image_path = image_list[images_recs - 1]
image_path_image = Image.open(image_path)

#resize_factor = 0.5  # Adjust this value for your desired percentage (e.g., 0.5 for 50% size)
#new_width = int(image_path_image.width * resize_factor)
#new_height = int(image_path_image.height * resize_factor)
#image_path_image = image_path_image .resize((new_width, new_height))

st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

with st.sidebar:
	# Display an image	
        #st.image(image_path).resize(newsize)
	st.image(image_path_image)


# st.markdown(
#     """
#     <style>
#         [data-testid=stSidebar]{
#             text-align: center;
#             display: block;
#             margin-left: auto;
# 	    margin-bottm:-75px;
#             margin-right: auto;
# 	    margin-top: -75px;     
#             width: 100%;
# 	    margin: 0;	         		
#         }
#     </style>
#     """, unsafe_allow_html=True
# )

with st.sidebar:
    st.markdown("""
        <style>
            [data-testid=stTextInput] {
                height: -5px;  # Adjust the height as needed
            }
        </style>
    """, unsafe_allow_html=True)

    text_input = st.text_input("Text Search: Enter animal. (Delete to use slider)", "", key = "text")
    openai_api_key = st.text_input('OpenAI API Key', "", type='password')


#text_input = st.sidebar.text_input("Text Search: Enter animal. (Delete to use slider)", "", key = "text")


#openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

####################################################################################################################################################

#@st.cache_resource
def plot_similar_images_new(image_path, text_input, number_of_images: int = 6):
	
	animal_embedding = model.encode(image_path)	

	if text_input:
		animal_embedding = model.encode(text_input)
		st.session_state.text_input = ""
	
	animal_embedding = torch.tensor(animal_embedding)

	################################################################################################################
	# Start of leveraging output of Qdrant	
	results = vector_db(client, animal_embedding)
	results = results[1:]

	result_image_type = results[0].payload['type'].capitalize()
	result_str = "You selected the following animal: " + result_image_type	
	
	if openai_api_key != "":		
		llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
		input_text = "Summarize in 100 words, the most interesting things about the following animal: " + result_str
		response = llm(input_text)
		st.write(response)
		#st.write("You selected the following animal: " + result_image_type + ". " + response)
	else:
		st.write(result_str  + ".")
		st.write("Enter an OpenAI API Key to learn more about " + result_image_type + 's!')

	grid_size = math.ceil(math.sqrt(number_of_images))
	axes = []
	fig = plt.figure(figsize=(20, 15))
        
	for i in range(len(results)):
		axes.append(fig.add_subplot(grid_size, grid_size, i + 1))
		plt.axis('off')
		#image_name = results[i].payload['image_name']
		image_path = results[i].payload['image_path']
		#image_type = results[i].payload['type']
		#image_score = results[i].score
		img = Image.open(image_path)
		img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
		plt.imshow(img_resized)
	#plt.title(f"Image {i}: {score}", fontsize=18)
	fig.tight_layout()
	fig.subplots_adjust(top=0.93)
	
	################################################################################################################
	
	# #Start of leveraging SentenceTransformer library
	# # Find the top 10 most similar images to the bear embedding.
	# most_similar_images = util.semantic_search(query_embeddings = animal_embedding, corpus_embeddings = img_emb_loaded, top_k = number_of_images)

	# # Create a list to store the results.
	# results = []

	# # Loop over the images in the most_similar_images variable.
	# for i in range(len(most_similar_images[0])):
	# 	# Get the image ID and score of the current image.
 #  		image_id = most_similar_images[0][i]['corpus_id']
 #  		image_score = most_similar_images[0][i]['score']

 #  		results.append([image_id, image_score])
	
	# grid_size = math.ceil(math.sqrt(number_of_images))
	# axes = []
	# fig = plt.figure(figsize=(20, 15))
        
	# for i in range(len(results)):
 #  		axes.append(fig.add_subplot(grid_size, grid_size, i + 1))
 #  		plt.axis('off')
 #  		image_number = results[i][0]
 #  		image_name = image_list[image_number]
 #  		score = results[i][1]
 #  		img = Image.open(image_name)
 #  		img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
 #  		plt.imshow(img_resized)
	# #plt.title(f"Image {i}: {score}", fontsize=18)
	# fig.tight_layout()
	# fig.subplots_adjust(top=0.93)

####################################################################################################################################################

if st.sidebar.button('Get Similar Images'):
	st.pyplot(plot_similar_images_new(image_path, text_input, number_of_images = 17))
	text_input = ""
	st.session_state.text_input = ""

#st.write(text_input)
####################################################################################################################################################	
