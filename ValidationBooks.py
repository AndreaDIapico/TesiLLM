import csv
import requests
import json
import re
from serpapi import GoogleSearch
from difflib import SequenceMatcher
from openai import OpenAI

def parse_authors(data):
    # If a book has a single author it is usually not in a list. This function standardizes the structure
    authors = []
    
    if isinstance(data["author"], list):
        # If "author" is a list, it has multiple authors
        for author_entry in data["author"]:
            author = author_entry["text"]
            authors.append(author)
    else:
        # If "author" is a dictionary, it has a single author
        author = data["author"]["text"]
        authors.append(author)
    
    return authors

def parse_authors_GPT(data):
    # If a book has a single author it is usually not in a list. This function standardizes the structure
    authors = []
    
    if isinstance(data, list):
        # If "author" is a list, it has multiple authors
        for author_entry in data:
            author = author_entry
            authors.append(author)
    else:
        # If "author" is a dictionary, it has a single author
        author = data
        authors.append(author)
    
    return authors

def similar(a, b):
    # Automatic similarity measure for strings
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()

with open("datasetBooks.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Initialize the OpenAI client, taking the OpenAI API Key from the environmental variables
client = OpenAI()

final_struct = []
counter = 0

for book in data:
    bookTitle = book['Book-Title']
    bookAuthors = book['Book-Author']
    bookYear = book['Year-Of-Publication']

    my_question = "Who and when wrote " + bookTitle

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      response_format={"type": "json_object"},
      messages=[
        {"role": "system", "content": "You are a skilled AI assistant, to every prompt you reply with a JSON where you list two fields exactly: the authors and the publication year"},
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
    GPTanswer = json.loads(modified_string)

    # Iterate through keys to find the 'authors' field. The GPT answer will sometimes modify the name of this field, generating exceptions otherwise
    for key, value in GPTanswer.items():
        if 'authors' in str(key).lower():  # Check if 'authors' is part of the key
            authours_value = value
            break
            
    # Iterate through keys to find the 'year' field. The GPT answer will sometimes modify the name of this field, generating exceptions otherwise
    for key, value in GPTanswer.items():
        if 'year' in str(key).lower():  # Check if 'year' is part of the key
            year_value = value
            break
            
    GPTanswer['authors'] = authours_value
    GPTanswer['authors'] = parse_authors_GPT(GPTanswer['authors'])
    GPTanswer['authors'] = " ".join(GPTanswer['authors'])
    GPTanswer['year'] = year_value
    GPTanswer['usage'] = response['usage']

    # Write the JSON data to file
    test_path = "GPTData4.json"
    with open(test_path, "w") as json_file:
        json_file.write(json.dumps(modified_string, indent=4))
    
    similarity = similar(bookAuthors, GPTanswer['authors'])
    
    # Output the results
    print(f"Prompt: {my_question}")
    print(f"Answer: {GPTanswer['authors']}\nYear: {GPTanswer['year']}\nTokens Used: {GPTanswer['usage']}")
    print(f"Correct Answer: {bookAuthors}\nYear: {bookYear}\nSimilarity: {similarity}")
    
    data_struct = {
        "title": bookTitle,
        "authors": GPTanswer['authors'],
        "year" : GPTanswer['year'],
        "correct_authors": bookAuthors,
        "correct_year": bookYear,
        "tokens": GPTanswer['usage'],
        "similarity": similarity
    }
    final_struct.append(data_struct)

    print(f"Finished analysing entry number {counter}")
    counter = counter + 1

    output_file_path = 'ValidationBooksResults2.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(final_struct, json_file, indent=2)