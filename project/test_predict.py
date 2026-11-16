import requests

url = 'http://localhost:9696/predict'

sample_book = {
    'weighted_average_rating': 4.46,
    'tag_name': 'fantasy',
    'authors': 'Patrick Rothfuss',
}

response = requests.post(url, json=sample_book).json()
print("API Response:", response)