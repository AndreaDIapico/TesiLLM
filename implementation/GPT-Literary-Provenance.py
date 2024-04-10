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
    
    if isinstance(data["authors"], list):
        # If "author" is a list, it has multiple authors
        for author_entry in data["authors"]:
            author = author_entry
            authors.append(author)
    else:
        # If "author" is a dictionary, it has a single author
        author = data["authors"]
        authors.append(author)
    
    return authors

def similar(a, b):
    # Automatic similarity measure for strings
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()

def search_DBLP(GPT):
    # Search book on DBLP
    GPTtitle = GPT['source']
    GPTauthors = GPT['authors']
    DBLPbooks = []
    DBLPwriters = []
    DBLP = []
    
    # Base URL for the DBLP API
    author_url = 'https://dblp.org/search/author/api'
    publ_url = 'https://dblp.org/search/publ/api'
    
    # Get the first 3 results
    query_url = f'{publ_url}?q={GPTtitle}&format=json&h=3'
    response = requests.get(query_url)
    
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()
        
        # Write the JSON data to file
        test_path = "DBLPData.json"
        with open(test_path, "w") as json_file:
            json_file.write(json.dumps(json_data, indent=4))
           
        try:
            subarray1 = json_data['result']['hits']['hit']
            
            # Extract the DBLP Data
            for element in subarray1:
                subarray2 = element['info']['authors']
                DBLPwriters.append(parse_authors(subarray2))
                DBLPbooks.append(element['info']['title'])
            
            # Unite the DBLP Data under DBLP variable
            for a, b in zip(DBLPbooks, DBLPwriters):
                DBLP.append([a, b])
            
            book_found = False
            
            # Compare DBLP with GPT
            for ele in DBLP:
                DBLPtitle, DBLPauthors = ele
                title_similarity = similar(GPTtitle, DBLPtitle)
                authors_similarity = similar(GPTauthors.replace(', ', ' '), " ".join(DBLPauthors))
                print("=" * 100)  
                print(f"GPT title = {GPTtitle}\nDBLP title = {DBLPtitle}\nSimilarity Index: {title_similarity}\n")
                print(f"GPT authors = {GPTauthors}\nDBLP authors = {DBLPauthors}\nSimilarity Index: {authors_similarity}")
                print("=" * 100)    
                if title_similarity > 0.7 and authors_similarity > 0.7:
                    book_found = True
            return book_found
        except KeyError:
            return False
        
    else:
        if response.status_code == 429:
            # Check if the "Retry-After" header is present in the response
            retry_after = response.headers.get('Retry-After')
            print(f'Failed to retrieve data. Status code: {response.status_code}\n')
            if retry_after:
                print(f'Retry-After header value: {retry_after}\n')
        else:        
            print(f'Failed to retrieve data. Status code: {response.status_code}\n')
        return False

def serpapi_search_and_parse(query_term):
    # Search on Google Scholar through the SerpAPI API
    GSchTitle = []
    GSchAuth = []
    GSchBooks = []
    
    params = {
      "engine": "google_scholar",
      "q": query_term,
      "api_key": "AUTH_KEY"
    }
    
    search = GoogleSearch(params)
    results = search.get_json()
    
    # Write the JSON data to file
    test_path = "GScholarData.json"
    with open(test_path, "w") as json_file:
        json_file.write(json.dumps(results, indent=4))     
    
    # Parse the results and obtain directly the Books, Authors pairs
    try:
        for hit in results['organic_results']:
            try: 
                sub = hit['publication_info']['authors']
                GSchTitle.append(hit['title'])
                authors = [entry["name"] for entry in sub]
                GSchAuth.append(authors)
            except KeyError:
                GSchTitle.append(hit['title'])
                input_string = hit['publication_info']['summary']
                matches = re.match(r"^(.*?)-", input_string)
                result = matches.group(1).strip()
                author_list = [author.strip() for author in result.split(',')]
                GSchAuth.append(author_list)
                
        for a, b in zip(GSchTitle, GSchAuth):
            GSchBooks.append([a, b])
        return GSchBooks
    except KeyError:
        return GSchBooks

def search_GScholar(GPTtoGS):
    # Search book on Google Scholar
    GPTtitle = GPTtoGS['source']
    GPTauthors = GPTtoGS['authors']
    found_books = serpapi_search_and_parse(GPTtitle) 
        
    # Split the string into individual names    
    names_list = [name.strip() for name in GPTauthors.split(',')]
    
    # Process each name to keep only the first letter of the names and the full surnames
    processed_names = []
    for name in names_list:
        name_parts = name.split(' ')
        first_letter = name_parts[0][0] if name_parts else ''
        surname = ' '.join(name_parts[1:])
        processed_names.append(f"{first_letter} {surname}")

    # Join the processed names back into a string
    GPTauthors = ' '.join(processed_names)

    # Check if a similar book with similar authors was found
    book_found = False
    for book in found_books:
        GStitle, GSauthors = book
        title_similarity = similar(GPTtitle, GStitle)   
        authors_similarity = similar(GPTauthors, ' '.join(GSauthors))
        print("=" * 100)  
        print(f"GPT title = {GPTtitle}\nGoogle Scholar title = {GStitle}\nSimilarity Index: {title_similarity}\n")
        print(f"GPT authors = {GPTauthors}\nGoogle Scholar authors = {GSauthors}\nSimilarity Index: {authors_similarity}")
        print("=" * 100)  
        if title_similarity > 0.7 and authors_similarity > 0.7:
            book_found = True
    return book_found

def search_GoogleBooks(book):
    # Search book on Google Scholar
    GPTtitle = book['source']
    GPTauthors = book['authors']
    # Define the base URL for the Google Books API
    base_url = "https://www.googleapis.com/books/v1/volumes"

    # Set up parameters for the API request
    params = {
        'q': f'intitle:{GPTtitle}',  # Search by book title
        'maxResults': 3,  # Limit the number of results to retrieve
    }

    try:
        # Make a GET request to the Google Books API
        response = requests.get(base_url, params=params)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Write the JSON data to file
            test_path = "GBooksData.json"
            with open(test_path, "w") as json_file:
                json_file.write(json.dumps(data, indent=4))
                
            # Process the retrieved book information
            if 'items' in data:
                book_found = False
                # Loop through the books in the response
                for item in data['items']:
                    book_info = item['volumeInfo']
                    GBtitle = book_info['title']
                    GBauthors = book_info.get('authors', ['N/A'])
                    title_similarity = similar(GPTtitle, GBtitle)
                    if isinstance(GPTauthors, list):
                        authors_similarity = similar(' '.join(GPTauthors), ' '.join(GBauthors))
                    else:
                        authors_similarity = similar(GPTauthors, ' '.join(GBauthors))
                    print("=" * 100)  
                    print(f"GPT title = {GPTtitle}\nGoogle Books title = {GBtitle}\nSimilarity Index: {title_similarity}\n")
                    print(f"GPT authors = {GPTauthors}\nGoogle Books authors = {GBauthors}\nSimilarity Index: {authors_similarity}")
                    print("=" * 100)  
                    if title_similarity > 0.7 and authors_similarity > 0.7:
                        book_found = True
                return book_found
            else:
                return False
        else:
            print(f"Request failed with status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False

# Initialize the OpenAI client, taking the OpenAI API Key from the environmental variables
client = OpenAI()
my_question = input("Enter your question: ")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={"type": "json_object"},
  messages=[
    {"role": "system", "content": "You are a reliable assistant that always provides their sources, to every prompt you reply with a JSON where you list exactly 3 fields reply, book source and authors"},
    {"role": "user", "content": my_question}
  ]
)

response = {
    "choices": completion.choices[0].message.content,
    "usage": completion.usage.total_tokens,
}

# Parse GPT answer to obtain a Reply Source Authors structure
input_string = response['choices']
modified_string = input_string.replace('\n', '').replace('\\', '')

# Write the JSON data to file
test_path = "GPTData.json"
with open(test_path, "w") as json_file:
    json_file.write(json.dumps(modified_string, indent=4))

GPTanswer = json.loads(modified_string)

# Iterate through keys to find the 'source' field. The GPT answer will sometimes modify the name of this field, generating exceptions otherwise
for key, value in GPTanswer.items():
    if 'source' in str(key).lower():  # Check if 'source' is part of the key
        source_value = value
        break
GPTanswer['source'] = source_value

# Iterate through keys to find the 'authors' field. The GPT answer will sometimes modify the name of this field, generating exceptions otherwise
for key, value in GPTanswer.items():
    if 'authors' in str(key).lower():  # Check if 'authors' is part of the key
        authours_value = value
        break
GPTanswer['authors'] = authours_value

GPTanswer['usage'] = response['usage']

# Output the results
print(f"Prompt: {my_question}")
print(f"Answer: {GPTanswer['reply']}\nSource: {GPTanswer['source']} by {GPTanswer['authors']}\nTokens Used: {GPTanswer['usage']}")

# Start the pipeline searching for the book
print("\nSearching now on DBLP")
FoundOnDBLP = search_DBLP(GPTanswer)
if not FoundOnDBLP:
    print("Not found on DBLP. Searching now on Google Books")
    FoundOnGoogleBooks = search_GoogleBooks(GPTanswer)
    if not FoundOnGoogleBooks:
        print("Not found on Google Books. Need to search on Google Scholar")
        FoundOnGoogleScholar = search_GScholar(GPTanswer)
        if not FoundOnGoogleScholar:
            print("The book was not found")
        else:
            print("The book was found on Google Scholar")
    else:
        print("The book was found on Google Books")
else:
    print("The book was found on DBLP")
