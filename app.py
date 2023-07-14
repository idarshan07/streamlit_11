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