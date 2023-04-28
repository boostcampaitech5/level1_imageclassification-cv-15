import streamlit as st
import cv2
import numpy as np
from PIL import Image
from augmentation import BaseAugmentation
import torch
from importlib import import_module

st.set_page_config(layout='wide', page_title="Mask Classification App")

MEAN=(0.548, 0.504, 0.479)
STD=(0.237, 0.247, 0.246)
RESIZE = (200, 200)
MODEL = "MobileNetV2"
MODEL_PATH = "weights/best.pth"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# num_classes = MaskBaseDataset.num_classes  # 18
CLASSES = 3 + 2 + 3
MASK_LABEL = {0 : "Mask",
              1 : "Incorrect Mask",
              2 : "Normal"}
GENDER_LABEL = {0 : "Male",
                1 : "Female"}
AGE_LABEL = {0 : "Young",
             1 : "Middle",
             2 : "Old"}

transform = BaseAugmentation(RESIZE, MEAN, STD)

@st.cache_data
def load_model(num_classes, device):
    model_cls = getattr(import_module("model"), MODEL)
    model = model_cls(
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    return model

model = load_model(num_classes=CLASSES, device=device)
model.eval()

def get_image(upload):
    image = np.array(Image.open(upload)).astype(np.uint8)
    st.subheader("Uploaded Image :sunglasses:")
    st.image(image)
    image = transform(image=image)['image'].unsqueeze(0)
    return image

def inference(image):
    with torch.no_grad():
        pred = model(image)
        (mask_out, gender_out, age_out) = torch.split(pred, [3, 2, 3], dim=1)
    
    mask = MASK_LABEL[mask_out.argmax(dim=-1).cpu().squeeze().item()]
    gender = GENDER_LABEL[gender_out.argmax(dim=-1).cpu().squeeze().item()]
    age = AGE_LABEL[age_out.argmax(dim=-1).cpu().squeeze().item()]
    return mask, gender, age

st.balloons()
st.snow()
st.write("## Mask Classification App :mask:")
st.sidebar.write("## *Upload image!!* :gear:")

my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    image = get_image(my_upload)
    with st.spinner(text='In progress'):
        mask, gender, age = inference(image=image)
        st.success('Inference Success! :white_check_mark:')    
    st.write(f"Mask : **{mask}**")
    st.write(f"Gender : **{gender}**")
    st.write(f"Age : **{age}**")

