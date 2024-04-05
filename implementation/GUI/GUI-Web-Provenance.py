import tkinter
import customtkinter
from PIL import Image
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import requests
import spacy
import string
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from numpy import dot
from numpy.linalg import norm
from pydantic import BaseModel, Field, validator
from scipy import spatial
from sentence_transformers import SentenceTransformer, util
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from typing import List

def similar(a, b):
    # Automatic similarity measure for strings
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()

def method_value_formatting(method, value):
    # Simple function to standardise the structure
    result = {"method": method, "value": value}
    return result

def word2vec_semantic_similarity(answer, html, shrink_value):
    print("word2vec\n")
    # Path to the pre-trained Google News Word2Vec model
    path_to_model = 'GoogleNews-vectors-negative300.bin.gz'

    # Load the pre-trained model
    word_vectors = KeyedVectors.load_word2vec_format(path_to_model, binary=True, limit=500000)
    
    # Custom similarity score that checks how many keywords match between the ChatGPT answer and the HTML text
    score = 0
    total = len(answer)
    totalhtml = len(html)
    matches = []
    for keyword in answer:
        # If the keyword matches exactly, add 1 to the score
        if keyword in html:
            score = score + 1
            matches.append(keyword)
            print(f"The word {keyword} was present in both documents")
            print("=" * 125)
        # Else if the keyword is present in the word2vec index
        elif keyword in word_vectors.key_to_index:
            # Find the 10 most similar words to our keyword
            # The array is ordered from the most similar to the least one and is made of word, similarity score pairs
            similar_array = word_vectors.most_similar(keyword, topn=10)
            print("=" * 125)
            print(f"The word {keyword} was not present, here are its similar words")
            print(similar_array)
            print("=" * 125)
            for similar_key in similar_array:
                word, similar = similar_key
                word = word.lower()
                # If the word is in the HTML keywords we add to the score the similarity between this word and our original keyword
                if word in html:
                    matches.append(keyword)
                    score = score + similar
                    print(f"Found a similar word in the crawled content!")
                    print(f"The most similar word to '{keyword}' was '{word}' with {similar} score") 
                    print("=" * 125)
                    # The first match will always be the most similar, so no need to check further
                    break
    print(f"Number of matches: {score}")
    print(f"Total number of keywords: {total}")
    print(f"Total number of keywords on the HTML: {totalhtml}")
    print(f"Shrink Term: {shrink_value}")
    print(f"Final score: {score/(shrink_value + total)}")
    print("The following words had no matches")
    print([word for word in answer if word not in matches])
    print("=" * 125)
    answer = f"Method: word2vec\tFinal score: {score/(shrink_value + total)}\nThe following words had no matches\t{[word for word in answer if word not in matches]}"
    # resw2v.configure(text=answer, wraplength=500)
    # app.update_idletasks()
    return method_value_formatting("word2vec", score/(shrink_value + total))

def wordnet_semantic_similarity(answer, html, shrink_value):
    print("WordNet\n")
    # Custom similarity score that checks how many keywords match between the ChatGPT answer and the HTML text
    score = 0
    total = len(answer)
    totalhtml = len(html)
    matches = []
    for keyword in answer:
        # If the keyword matches exactly, add 1 to the score
        if keyword in html:        
            score = score + 1
            matches.append(keyword)
            print(f"The word {keyword} was present in both documents")
            print("=" * 125)
        # Else if the keyword has synonyms in the WordNet dictionary    
        elif wn.synonyms(keyword):
            print("=" * 125)
            print(f"The word {keyword} was not present, here are its synonyms")
            print(wn.synonyms(keyword))
            print("=" * 125)
            top_simi = 0
            chosen_word = ""
            # Each word has multiple arrays of synonyms divided into semantic groups
            for array in wn.synonyms(keyword):
                # Explore all the arrays
                for word in array:
                    temp1 = wn.synsets(word)
                    temp2 = wn.synsets(keyword)
                    # If a word similar to our keyword is present inside the HTML keywords calculate the similarity
                    if word in html:
                        simi = temp1[0].path_similarity(temp2[0])
                        print(f"Found a similar word in the crawled content!")
                        print(f"Similarity between '{keyword}' and '{word}': {simi}")
                        # If it's the best result so far, save it
                        if simi > top_simi:
                            top_simi = simi
                            chosen_word = word
            # After exploring all the array, add the most similar score
            if chosen_word:
                matches.append(keyword)
                score = score + top_simi
                print(f"The most similar word to '{keyword}' was '{chosen_word}' with {top_simi} score") 
                print("=" * 125)
    print(f"Number of matches: {score}")
    print(f"Total number of keywords: {total}")
    print(f"Total number of keywords on the HTML: {totalhtml}")
    print(f"Shrink Term: {shrink_value}")
    print(f"Final score: {score/(shrink_value + total)}")
    print("The following words had no matches")
    print([word for word in answer if word not in matches])
    print("=" * 125)
    answer = f"Method: WordNet\tFinal score: {score/(shrink_value + total)}\nThe following words had no matches\t{[word for word in answer if word not in matches]}"
    # reswn.configure(text=answer, wraplength=500)
    # app.update_idletasks()
    return method_value_formatting("WordNet", score/(shrink_value + total))

def wmd_semantic_similarity(answer, html):
    print("Word Mover's Distance\n")
    # Dissimilarity metric, returns a number, the lower, the better
    path_to_model = 'GoogleNews-vectors-negative300.bin.gz'
    word_vectors = KeyedVectors.load_word2vec_format(path_to_model, binary=True, limit=500000)

    wmd_similarity = WmdSimilarity([answer], word_vectors)
    distance = wmd_similarity[html]
    print(f"Word Mover's Distance between the documents: {distance[0]:.4f}")
    answer = f"Method: Word Mover Distance\tFinal score: {distance[0]}"
    # reswmd.configure(text=answer, wraplength=500)
    # app.update_idletasks()
    return method_value_formatting("Word Movers Distance", distance[0])

def find_closest_embeddings(embedding, embeddings_dict):
    # Return the closest embeddings from a given word
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def word_embedding_similarity(word1, word2, embeddings_dict):
    # Returns the similarity between two words
    vec1 = embeddings_dict[word1]
    vec2 = embeddings_dict[word2]
    similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity

def glove_semantic_similarity(answer, html, shrink_value):
    print("GloVe\n")
    # Custom similarity score that checks how many keywords match between the ChatGPT answer and the HTML text
    embeddings_dict = {}
    with open("glove.6B.300d.txt", 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector    

    score = 0
    total = len(answer)
    totalhtml = len(html)
    matches = []
    for keyword in answer:
        # If the keyword matches exactly, add 1 to the score
        if keyword in html:
            score = score + 1
            matches.append(keyword)
            print(f"The word {keyword} was present in both documents")
            print("=" * 125)
        # Else if the word is in the embeddings dictionary
        elif keyword in embeddings_dict:
            # Find the closest words
            new_key = find_closest_embeddings(embeddings_dict[keyword], embeddings_dict)[1:6]
            print("=" * 125)
            print(f"The word {keyword} was not present, here are its closest words")
            print(new_key)
            print("=" * 125)
            top_simi = 0
            chosen_word = ""
            # Find the most similar word that is included in the HTML
            for key in new_key:
                if key in html:
                    similarity = word_embedding_similarity(key, keyword, embeddings_dict)
                    print(f"Found a similar word in the crawled content!")
                    print(f"Similarity between '{keyword}' and '{key}': {similarity:.4f}")
                    if similarity > top_simi:
                        top_simi = similarity
                        chosen_word = key
            # After iterating all the array, add the most similar score to the total score               
            if chosen_word:
                score = score + top_simi
                matches.append(keyword)
                print(f"The most similar word to '{keyword}' was '{chosen_word}' with {top_simi} score") 
                print("=" * 125)
    print(f"Number of matches: {score}")
    print(f"Total number of keywords: {total}")
    print(f"Total number of keywords on the HTML: {totalhtml}")
    print(f"Shrink Term: {shrink_value}")
    print(f"Final score: {score/(shrink_value + total)}")
    print("The following words had no matches")
    print([word for word in answer if word not in matches])
    print("=" * 125)
    answer = f"Method: GloVe\tFinal score: {score/(shrink_value + total)}\nThe following words had no matches\t{[word for word in answer if word not in matches]}"
    # resglove.configure(text=answer, wraplength=500)  
    # app.update_idletasks()
    return method_value_formatting("GloVe", score/(shrink_value + total))

def msmarco_semantic_similarity(answer, html):
    print("MSMarco\n")
    # Similarity measure indicating the similarity between two documents, ranges from -1 to 1, with 1 meaning similar and -1 completely dissimilar
    # Load a pre-trained model
    model_name = 'msmarco-distilbert-base-v4'
    model = SentenceTransformer(model_name)

    # Encode sentences to embeddings
    embeddings = model.encode([answer, html], convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    print(f"Cosine Similarity using MSMarco: {cos_sim.item()}")
    answer = f"Method: MSMarco\tFinal score: {cos_sim.item()}"
    # resmsm.configure(text=answer, wraplength=500)
    # app.update_idletasks()    
    return method_value_formatting("MSMarco", cos_sim.item())

def check_link_contents(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Print the content of the response (HTML content for web pages)
            return response.text
        else:
            print(f"Request failed with status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")

def split_link(link):
    # Define the regex pattern to extract domain and path
    pattern = r'(https?://[^/]+)(/.*)?'

    # Use regex to match the pattern
    match = re.match(pattern, link)

    if match:
        domain = match.group(1)
        path = match.group(2) if match.group(2) else '/'
        return domain, path

    return None, None

def tokenize_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Stop words removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Punctuation removal
    tokens = [token for token in tokens if token not in string.punctuation]
    # Digit removal
    tokens = [token for token in tokens if not token.isdigit()]
    # Second layers of removal with spaCy
    en = spacy.load('en_core_web_sm')
    sw_spacy = en.Defaults.stop_words
    tokens = [token for token in tokens if token.lower() not in sw_spacy]
    return ' '.join(tokens)

def tokenize_html(content):
    # Create a BeautifulSoup object
    soup = BeautifulSoup(content, 'html.parser')
    # Get the text without HTML tags
    text_without_tags = soup.get_text()
    text_without_tags = text_without_tags.strip()
    filtered_text = tokenize_text(text_without_tags)
    test_path = "rescrawling2.json"
    with open(test_path, "w", encoding='utf-8') as json_file:
        json_file.write(filtered_text) 
    return filtered_text

def semantic_similarity(answer, html, shrink_value):
    result_array = []
    
    # print("=" * 125)
    # print("Evaluating similarity between the GPT answer and the HTML of the crawled page")
    # print("=" * 125)
    
    crawled_page_content = html.split()
    short_answer = answer.split()
    
    # print("=" * 125)
    # print("1st Method: words2vec")
    # print("=" * 125)
    result_array.append(word2vec_semantic_similarity(short_answer, crawled_page_content, shrink_value))
    
    # print("=" * 125)
    # print("2nd Method: WordNet")
    # print("=" * 125)
    result_array.append(wordnet_semantic_similarity(short_answer, crawled_page_content, shrink_value))

    # print("=" * 125)
    # print("3rd Method: Word Mover's Distance")
    # print("=" * 125)
    result_array.append(wmd_semantic_similarity(short_answer, crawled_page_content))

    # print("=" * 125)
    # print("4th Method: GloVe")
    # print("=" * 125)
    result_array.append(glove_semantic_similarity(short_answer, crawled_page_content, shrink_value))  

    # print("=" * 125)
    # print("5th Method: MSMarco")
    # print("=" * 125)
    result_array.append(msmarco_semantic_similarity(short_answer, crawled_page_content))
    
    temp_final = 0
    flag = 1
    for element in result_array:
        quality = ['word2vec', 'WordNet', 'GloVe']
        if element['method'] in quality:
            temp_final = temp_final + element['value']
            if element['value'] < 0.6:
                flag = 0 
    temp_final = temp_final / 3
    temp_final = round(temp_final, 2)
    
    if flag == 1:
        my_text = f"The answer has high similarity with the provenance provided\nFinal Score: {temp_final}"
        resw2v.configure(text=my_text, wraplength=500, text_color='green', font=my_font)
        reswn.configure(image=img1)
        app.update_idletasks()
    else:
        my_text = f"The answer does not have high similarity with the provenance provided\nFinal Score: {temp_final}"
        resw2v.configure(text=my_text, wraplength=500, text_color='red', font=my_font)
        reswn.configure(image=img2)
        app.update_idletasks() 
    
    return result_array

def term_frequency_thresholding(document, custom_threshold):
    vectorizer = TfidfVectorizer()

    # Fit and transform the document
    tfidf_matrix = vectorizer.fit_transform([document])
    # Set a term frequency threshold (adjust as needed)
    threshold = custom_threshold

    # Get the feature names (terms)
    features = vectorizer.get_feature_names_out()

    # Get the TF-IDF values and associated terms above the threshold
    above_threshold_terms = [
        term for tfidf, term in zip(tfidf_matrix[0].toarray()[0], features) if tfidf >= threshold
    ]
    
    return ' '.join(above_threshold_terms)

def get_data_from_json(cleaned_json):
    # Removing backslashes from the JSON string
    cleaned_json = cleaned_json.replace('\")', ')').replace('\n', '').replace('\\"', '"').replace('\\\"', '"')
    data = json.loads(cleaned_json)
    # Extract only the "content" fields from the messages
    content_list = [message['content'] for message in data['chat_memory']['messages']]
    saved_response = content_list[1]
    # Save the response for later, so that it can be searched on Google
    saved_response_to_google = saved_response
    # Tokenize the GPT response
    saved_response = tokenize_text(saved_response)
    # Perform TF Thresholding on the tokenized text
    # saved_response = term_frequency_thresholding(saved_response, 0.1)
    # print("=" * 125)
    # print("***** GPT Keywords *****")
    # print(saved_response)
    # print("=" * 125)
    GPTdomain = ""
    GPTpath = ""
    final_results = []
    try:
        # Regular expression pattern to match URLs
        url_pattern = r'https?://\S+'
        # Find all URLs in the string using regex
        urls = re.findall(url_pattern, content_list[3])
        for url in urls:
            # Find trailing characters, GPT often includes links inside brackets
            pattern = r'[)\]}]$'
            match = re.search(pattern, url)
            if match:
                url = url[:-1]
            GPTdomain, GPTpath = split_link(url)
            try:
                # Obtain the HTML text
                htmlpage = check_link_contents(url)
                # Clean the HTML from tags and tokenize it
                htmlpage = tokenize_html(htmlpage)
                # TF Thresholding
                # htmlpage = term_frequency_thresholding(htmlpage, 0.02)
                # print("=" * 125)
                # print("***** HTML Keywords *****")
                # print(htmlpage)
                # print("=" * 125)
                # Save the results
                test_path = "rescrawling.json"
                with open(test_path, "w", encoding='utf-8') as json_file:
                    json_file.write(htmlpage) 
                GPTlink = GPTdomain + GPTpath
                final_results.append({"link": GPTlink, "similarities": semantic_similarity(saved_response, htmlpage, 2)})
            except TypeError:
                pass
    except IndexError:
        pass

# Initialize the OpenAI client, taking the OpenAI API Key from the environmental variables        
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, max_tokens=256)

# Initialize the Conversation Memory, so we can keep track of previous messages
memory = ConversationBufferMemory()
template = """The following is a conversation between a human and an AI. 
The AI is brief and likes to provide a link to their reference only if prompted to.

Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=prompt,
    llm=llm, 
    memory=memory
)

def gpt_interaction():
    my_question = textbox.get()
    answer = conversation.run(input=my_question)
    yourMessageLabel.configure(text=my_question, wraplength=500)
    myinput.set("")
    responseLabel.configure(text=answer, wraplength=500)
    get_data_from_json(conversation.memory.json())

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("720x480")
app.title("Link provenance analysis")
my_font = customtkinter.CTkFont(family="", size=18, weight="bold")

title = customtkinter.CTkLabel(app, text="Your question")
title.pack()

myinput = tkinter.StringVar()
textbox = customtkinter.CTkEntry(app, width=350, height=35, textvariable=myinput)
textbox.pack()

start = customtkinter.CTkButton(app, text="Send", command=gpt_interaction)
start.pack(padx=20, pady=5)

yourMessageLabel = customtkinter.CTkLabel(app, text="", wraplength=500)
yourMessageLabel.pack(padx=20, pady=5)

responseLabel = customtkinter.CTkLabel(app, text="", wraplength=500)
responseLabel.pack(padx=20, pady=5)

resw2v = customtkinter.CTkLabel(app, text="", wraplength=1000)
resw2v.pack(padx=20, pady=10)

img1 = customtkinter.CTkImage(light_image=Image.open('GreenCheck.png'), size=(50,50))
img2 = customtkinter.CTkImage(light_image=Image.open('RedCross.png'), size=(50,50))

reswn = customtkinter.CTkLabel(app, text="", wraplength=1000)
reswn.pack(padx=20, pady=5)

reswmd = customtkinter.CTkLabel(app, text="", wraplength=1000)
reswmd.pack(padx=20, pady=5)

resglove = customtkinter.CTkLabel(app, text="", wraplength=1000)
resglove.pack(padx=20, pady=5)

resmsm = customtkinter.CTkLabel(app, text="", wraplength=1000)
resmsm.pack(padx=20, pady=5)

app.mainloop()
