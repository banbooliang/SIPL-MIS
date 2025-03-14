import os
import clip
import torch


## PAOT

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus',
             'Liver', 'Stomach', 'Aorta', 'Postcava', 'Portal Vein and Splenic Vein',
            'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland']  # BTCV dataset

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = calip.load('ViT-B/32', device)
model = torch.jit.load('./RN101.pt', map_location="cuda:0").eval()

text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding.pth')

