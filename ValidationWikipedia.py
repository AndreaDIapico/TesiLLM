import json
import math
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
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
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

# 6458670 Dataset size
# n = [z^2 * p * (1-p)] / e^2
# z = 95% confidence level = 1.96
# p = 0.5 conservative estimate
# e = 0.01 / 0.03 / 0.05
# Result: n = 9604 / 1068 / 385 -> Varies from e

def method_value_formatting(method, value):
    # Simple function to standardise the structure
    result = {"method": method, "value": value}
    return result

def word2vec_semantic_similarity(answer, html, shrink_value):
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
            # print(f"The word {keyword} was present in both documents")
            # print("=" * 125)
        # Else if the keyword is present in the word2vec index
        elif keyword in word_vectors.key_to_index:
            # Find the 10 most similar words to our keyword
            # The array is ordered from the most similar to the least one and is made of word, similarity score pairs
            similar_array = word_vectors.most_similar(keyword, topn=10)
            for similar_key in similar_array:
                word, similar = similar_key
                word = word.lower()
                # If the word is in the HTML keywords we add to the score the similarity between this word and our original keyword
                if word in html:
                    matches.append(keyword)
                    score = score + similar
                    # print(f"Found a similar word in the crawled content!")
                    # print(f"The most similar word to '{keyword}' was '{word}' with {similar} score") 
                    # print("=" * 125)
                    # The first match will always be the most similar, so no need to check further
                    break
    # print(f"Number of matches: {score}")
    # print(f"Total number of keywords: {total}")
    # print(f"Total number of keywords on the HTML: {totalhtml}")
    # print(f"Shrink Term: {shrink_value}")
    # print(f"Final score: {score/(shrink_value + total)}")
    # print("The following words had no matches")
    # print([word for word in answer if word not in matches])
    return method_value_formatting("word2vec", score/(shrink_value + total))

def wordnet_semantic_similarity(answer, html, shrink_value):
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
            # print(f"The word {keyword} was present in both documents")
            # print("=" * 125)
        # Else if the keyword has synonyms in the WordNet dictionary    
        elif wn.synonyms(keyword):
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
                        # print(f"Found a similar word in the crawled content!")
                        # print(f"Similarity between '{keyword}' and '{word}': {simi}")
                        # If it's the best result so far, save it
                        if simi > top_simi:
                            top_simi = simi
                            chosen_word = word
            # After exploring all the array, add the most similar score
            if chosen_word:
                matches.append(keyword)
                score = score + top_simi
                # print(f"The most similar word to '{keyword}' was '{chosen_word}' with {top_simi} score") 
                # print("=" * 125)
    # print(f"Number of matches: {score}")
    # print(f"Total number of keywords: {total}")
    # print(f"Total number of keywords on the HTML: {totalhtml}")
    # print(f"Shrink Term: {shrink_value}")
    # print(f"Final score: {score/(shrink_value + total)}")
    # print("The following words had no matches")
    # print([word for word in answer if word not in matches])
    return method_value_formatting("WordNet", score/(shrink_value + total))

def wmd_semantic_similarity(answer, html):
    # Dissimilarity metric, returns a number, the lower, the better
    path_to_model = 'GoogleNews-vectors-negative300.bin.gz'
    word_vectors = KeyedVectors.load_word2vec_format(path_to_model, binary=True, limit=500000)

    wmd_similarity = WmdSimilarity([answer], word_vectors)
    distance = wmd_similarity[html]
    # print(f"Word Mover's Distance between the documents: {distance[0]:.4f}")
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
            # print(f"The word {keyword} was present in both documents")
            # print("=" * 125)
        # Else if the word is in the embeddings dictionary
        elif keyword in embeddings_dict:
            # Find the closest words
            new_key = find_closest_embeddings(embeddings_dict[keyword], embeddings_dict)[1:6]
            top_simi = 0
            chosen_word = ""
            # Find the most similar word that is included in the HTML
            for key in new_key:
                if key in html:
                    similarity = word_embedding_similarity(key, keyword, embeddings_dict)
                    # print(f"Found a similar word in the crawled content!")
                    # print(f"Similarity between '{keyword}' and '{key}': {similarity:.4f}")
                    if similarity > top_simi:
                        top_simi = similarity
                        chosen_word = key
            # After iterating all the array, add the most similar score to the total score               
            if chosen_word:
                score = score + top_simi
                matches.append(keyword)
                # print(f"The most similar word to '{keyword}' was '{chosen_word}' with {top_simi} score") 
                # print("=" * 125)
    # print(f"Number of matches: {score}")
    # print(f"Total number of keywords: {total}")
    # print(f"Total number of keywords on the HTML: {totalhtml}")
    # print(f"Shrink Term: {shrink_value}")
    # print(f"Final score: {score/(shrink_value + total)}")
    # print("The following words had no matches")
    # print([word for word in answer if word not in matches])
    return method_value_formatting("GloVe", score/(shrink_value + total))

def msmarco_semantic_similarity(answer, html):
    # Similarity measure indicating the similarity between two documents, ranges from -1 to 1, with 1 meaning similar and -1 completely dissimilar
    # Load a pre-trained model
    model_name = 'msmarco-distilbert-base-v4'
    model = SentenceTransformer(model_name)

    # Encode sentences to embeddings
    embeddings = model.encode([answer, html], convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    # print(f"Cosine Similarity using MSMarco: {cos_sim.item()}")
    return method_value_formatting("MSMarco", cos_sim.item())

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
    
    return result_array

def tokenize_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
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

with open("datasetWikipedia2.json", "r", encoding="utf-8") as file:
    data = json.load(file)

final_struct = []
counter = 0
 
for element in data: 
    # Initialize the OpenAI client, taking the OpenAI API Key from the environmental variables
    client = OpenAI()
    my_question = "Tell me about " + element['title']
    shrink_value = len(element['title'].split())

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      response_format={"type": "json_object"},
      messages=[
        {"role": "system", "content": "You are a reliable and accurate AI assistant that talks at length about the asked topic, to every prompt you reply with a JSON where you list exactly 2 fields reply and your link source"},
        {"role": "user", "content": my_question}
      ]
    )

    response = {
        "choices": completion.choices[0].message.content,
        "usage": completion.usage.total_tokens,
    }

    # Parse GPT answer to obtain a Reply Source Authors structure
    input_string = response['choices']
    modified_string = input_string.replace('\")', ')').replace('\n', '').replace('\\"', '"')

    # Write the JSON data to file
    test_path = "GPTData2.json"
    with open(test_path, "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(modified_string, indent=4))

    GPTanswer = json.loads(modified_string)

    # Iterate through keys to find the 'source' field. The GPT answer will sometimes modify the name of this field, generating exceptions otherwise
    for key, value in GPTanswer.items():
        if 'source' in str(key).lower():  # Check if 'source' is part of the key
            source_value = value
            break
    GPTanswer['source'] = source_value

    GPTanswer['usage'] = response['usage']

    # Output the results
    print(f"Prompt: {my_question}")
    print(f"Answer: {GPTanswer['reply']}\nSource: {GPTanswer['source']}\nTokens Used: {GPTanswer['usage']}")

    GPT = tokenize_text(GPTanswer['reply'])
    Wikipedia = tokenize_text(element['text'])
    results = semantic_similarity(GPT, Wikipedia, shrink_value)
    data_struct = {
        "answer": GPTanswer['reply'],
        "source": GPTanswer['source'],
        "tokens": GPTanswer['usage'],
        "results": results
    }
    final_struct.append(data_struct)
    print(f"Finished analysing entry number {counter}")
    counter = counter + 1

    output_file_path = 'ValidationWikipediaResults4.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(final_struct, json_file, indent=2)