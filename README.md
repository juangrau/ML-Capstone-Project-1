# Data Talks Club - Machine Learning Zoomcamp
# Capstone Project 1

## Satellite Terrain Image Classification

### Introduction

Satellite imagery provides a unique perspective on our planet, capturing vast extensions of land and sea with unparalleled detail. However, manually analyzing these images for specific features like land cover (forests, urban areas, water bodies, industrial areas), crop types, or disaster zones is a time-consuming and labor-intensive task. This project aims to leverage the power of Convolutional Neural Networks (CNNs), a class of deep learning algorithms particularly adept at image analysis, to automate the classification of satellite images.

#### Potential Use Cases

**Precision Agriculture:**
- Crop Monitoring: Classify different crop types to optimize irrigation, fertilization, and pest control.
- Disease Detection: Identify diseased crops or areas affected by drought.

**Environmental Monitoring:**
- Deforestation Detection: Track deforestation rates and identify areas of illegal logging.
- Land Cover Mapping: Create accurate and up-to-date maps of land cover types (forests, grasslands, urban areas).
- Coastal Erosion Monitoring: Assess coastal erosion rates and predict potential risks.

**Disaster Management:**
- Flood Detection: Identify flooded areas and assess the extent of damage.
- Wildfire Detection: Early detection of wildfires to aid in rapid response and containment efforts.
- Earthquake Damage Assessment: Rapidly assess the extent of damage to infrastructure and buildings.

**Urban Planning:**
- Urban Growth Monitoring: Track urban expansion and identify areas of high population density.
- Infrastructure Mapping: Map roads, buildings, and other infrastructure for urban planning and development.

This project has the potential to significantly impact various fields by providing valuable insights from satellite imagery in a more efficient and accurate manner.

### Core objective of this project

1. **Data aquisition and preparation**. In this case we are using the EuroSAT dataset. The dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images. A more detailed description of this dataset can be found [here]('https://github.com/phelber/eurosat')

2. **Model Selection**. In this case we'll use Convolutional Neural Network as the ML model. As the framework for training and tunning the model we'll use Tensorflow and Keras.

3. **Model Training and Tunning**. We'll use parameters like learning rate, dropout rate and data augmentation as the parameters to tune our model.

4. **Prediction**. Utilize the trained model to classify satellite images based on the 10 categories available on the dataset.

5. **Containerization**. To be able to reproduce this result, the model will be placed on a Docker container. 

6. **Model deployment**. Finally, now that the model was placed on a Docker container, the last step of the project will be to setup a serverless deployment environment on AWS and run the project from there.

### Availabe files on the repository


### How to run this project

