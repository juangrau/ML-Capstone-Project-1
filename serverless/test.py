import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://github.com/juangrau/ML-Capstone-Project-1/blob/main/AnnualCrop_1000.jpg?raw=true'}

result = requests.post(url, json=data).json()
print(result)