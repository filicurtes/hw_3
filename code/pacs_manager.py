import pandas as pd 
import numpy as np 
import os 
from torchvision.datasets import ImageFolder
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img=Image.open(f)
        return img.convert('RGB')
def pacs:
    cwd=os.getcwd()
    os.chdir('PACS')
    