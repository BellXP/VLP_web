import torch
import requests
from PIL import Image
from peng_utils import TestBlip2, TestMiniGPT4, TestMplugOwl, TestMultimodelGPT, TestOtter, TestFlamingo

device = torch.device('cuda:7')
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
raw_image = Image.open('merlion.png').convert('RGB')
question = "which city is this?"


test_model = TestOtter()
print(test_model.generate(question, raw_image, device))
print(test_model.generate(question, raw_image))