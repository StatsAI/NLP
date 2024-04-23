# © Stats AI LLC 2023. All Rights Reserved. 
# No part of this code may be used or reproduced without express permission.

import streamlit as st
#from tqdm import tqdm
# import numpy as np
# from torchvision import transforms
# import torch
# #from torch.autograd import Variable
# import os
# import math
# import time
# import uuid
# from qdrant_client import QdrantClient
# from qdrant_client.http import models as rest
# import requests
# import zipfile
# import json

# from langchain_openai import OpenAI
# from typing import List
# from sentence_transformers import SentenceTransformer, util

# #from IPython.display import display
# import matplotlib.pyplot as plt
# from PIL import Image
# from PIL import ImageOps
# import pandas as pd

from unstructured.partition.html import partition_html
import requests

st.set_option('deprecation.showPyplotGlobalUse', False)

####################################################################################################################################################

#from unstructured.partition.html import partition_html
#import requests


#links

####################################################################################################################################################

# url = "https://github.com/StatsAI/streamlit_image_search_db/releases/download/image_search_assets/archive.zip"
# download_and_unzip(url)

# image_list, img_emb_loaded, img_type = load_assets()

# model = load_model()

# client = create_vector_db_input(img_emb_loaded)

####################################################################################################################################################

# logo = Image.open('images/picture.png')
# #newsize = (95, 95)
# #logo = logo.resize(newsize)

# st.markdown(
#     """
#     <style>
#         [data-testid=stSidebar] [data-testid=stImage]{
#             text-align: center;
#             display: block;
#             margin-left: auto;
#             margin-right: auto;
# 	    margin-top: -75px;
#             width: 100%;
# 	    #margin: 0;	         		
#         }
#     </style>
#     """, unsafe_allow_html=True
# )

# with st.sidebar:
#     st.image(logo)

# st.markdown("""
#         <style>
#                .block-container {
# 		    padding-top: 0;
#                 }
#         </style>
#         """, unsafe_allow_html=True)

# #st.write('')
# #st.write('')
st.title("CNN News Summarization via Unstructured + LangChain + ChromaDB + OpenAI")
st.write("This app enables a user to automatically search/summarize the latest CNN articles!")

# images_recs = st.sidebar.slider(label = 'Image Search: Select an animal using the slider', min_value = 1,
#                           max_value = 5400,
#                           value = 1859,
#                           step = 1)

# image_path = image_list[images_recs - 1]
# image_path_image = Image.open(image_path)

# #resize_factor = 0.5  # Adjust this value for your desired percentage (e.g., 0.5 for 50% size)
# #new_width = int(image_path_image.width * resize_factor)
# #new_height = int(image_path_image.height * resize_factor)
# #image_path_image = image_path_image .resize((new_width, new_height))

# st.sidebar.write('')
# st.sidebar.write('')
# st.sidebar.write('')
# st.sidebar.write('')

# with st.sidebar:
# 	# Display an image	
#         #st.image(image_path).resize(newsize)
# 	st.image(image_path_image)


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
    # st.markdown("""
    #     <style>
    #         [data-testid=stTextInput] {
    #             height: -5px;  # Adjust the height as needed
    #         }
    #     </style>
    # """, unsafe_allow_html=True)

    text_input = st.text_input("CNN Topic Search: Enter topic you want to summarize articles", "", key = "text")
    openai_api_key = st.text_input('OpenAI API Key', "", type='password')

with st.sidebar:
    data_source = st.radio(
    "Select your data source",
    ["GitHub", "Web"],
    index=None,)

####################################################################################################################################################

#@st.cache_resource
def import_html_from_github():

	#1. Get the links for the latest articles from an HTML file (Option 1).
	#2. Use the Unstructured document loader in Langchain to load the files.
	#3. Create embeddings for each file using OpenAIEmbeddings.
	#4. Store the embeddings in Chroma DB.
	#5. Query Chroma DB to return relevant articles.
	#6. Summarize the relevant articles using LangChain OpenAI integration.
	
	url = "https://github.com/StatsAI/NLP/blob/main/Breaking%20News%2C%20Latest%20News%20and%20Videos%20_%20CNN.html"

	try:
    		response = requests.get(url, allow_redirects=True)
    		response.raise_for_status()  # Raise error if download fails
	except requests.exceptions.RequestException as e:
    		print(f"Error downloading HTML: {e}")
    		exit(1)

	# Save the content to a file
	with open("downloaded_html.html", "wb") as f:
	    f.write(response.content)

	elements = partition_html(filename='downloaded_html.html')
	elements = elements[3].links
	
	links = []
	cnn_lite_url = "https://lite.cnn.com/"
	
	for element in elements:
	  try:
	    if element["url"][3:-2]:
	      relative_link = element["url"][3:-2]
	      links.append(f"{cnn_lite_url}{relative_link}")
	  except IndexError:
	    # Handle the case where the "url" key doesn't exist or the index is out of range
	    continue

	return links


def import_html_from_web():

	#1. Get the links for the latest articles from an HTML file (Option 1).
	#2. Use the Unstructured document loader in Langchain to load the files.
	#3. Create embeddings for each file using OpenAIEmbeddings.
	#4. Store the embeddings in Chroma DB.
	#5. Query Chroma DB to return relevant articles.
	#6. Summarize the relevant articles using LangChain OpenAI integration.
	
	cnn_lite_url = "https://lite.cnn.com/"

	elements = partition_html(url=cnn_lite_url)
	links = []
	
	for element in elements:
	  try:
	    if element.links[0]["url"][1:]:
	      relative_link = element.links[0]["url"][1:]
	      links.append(f"{cnn_lite_url}{relative_link}")
	  except IndexError:
	    # Handle the case where the "url" key doesn't exist or the index is out of range
	    continue
	
	return links, cnn_lite_url
	
####################################################################################################################################################

if st.sidebar.button('Summarize relevant docs'):

	if data_source == "Github":
		result = import_html_from_github()
		st.write(result.values())
		
	if data_source == "Web":
		result,url = import_html_from_web()
		result = result[:-2]
		st.write(url)
		st.write("Here are the articles!")

		for element in result:
			st.write(element)
		
	#st.pyplot(plot_similar_images_new(image_path, text_input, number_of_images = 17))
	#text_input = ""
	#st.session_state.text_input = ""

#st.write(text_input)
####################################################################################################################################################	
