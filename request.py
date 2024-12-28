'''
*************** SCRIPT DESCRIPTION *****************

The idea of this script is to test the flask service
that makes the prediction based an url of an image of 
terrain satellite picture

Below you will find additional images to try

****************************************************
'''
import requests

'''
Other Images are here:
https://github.com/juangrau/ML-Capstone-Project-1/blob/main/AnnualCrop_1004.jpg?raw=true
https://github.com/juangrau/ML-Capstone-Project-1/blob/main/Residential_1000.jpg?raw=true
https://github.com/juangrau/ML-Capstone-Project-1/blob/main/River_1002.jpg?raw=true

'''


data = {
    'url': 'https://github.com/juangrau/ML-Capstone-Project-1/blob/main/AnnualCrop_1000.jpg?raw=true'
    }
#url = 'http://172.25.212.32:9696/predict'
url = 'http://127.0.0.1:9696/predict'

result = requests.post(url, json=data).json()
print(result)


