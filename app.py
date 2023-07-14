import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import wget
import os

# Load the pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

def load_labels():
    labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    labels_file = os.path.basename(labels_path)
    if not os.path.exists(labels_file):
        wget.download(labels_path)
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories
    
# Transform the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Streamlit App
def main():
    st.title('Image Classification with ResNet-18')
    st.write('Upload an image and the app will predict its class.')


    # Image upload
    uploaded_image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    # Define the ImageNet class labels
    class_labels = load_labels()