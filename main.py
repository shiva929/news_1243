import streamlit as st
import pandas as pd
import layoutparser as lp
import pdf2image
import pytesseract
from PIL import Image
from functools import lru_cache
import re
import tempfile
import os
import requests
from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_colwidth', None)
st.title("The Digital Times")
np = st.selectbox(label = 'Select Your Newspaper', index = None, options = ['Economic Times', 'Times of India'])

Image.LINEAR = Image.BILINEAR

@lru_cache(maxsize=1)
def load_model():
    model_path = '/root/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth'  # Updated path
    return lp.models.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        model_path=model_path,  # Specify the downloaded model path
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.3
        ],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )

def is_valid_headline(headline, page_width, page_height):
    centroid_y = (headline.y_1 + headline.y_2) / 2  # Compute Y-coordinate of the centroid
    return (
        headline.height > 10  # Further lowered threshold to capture smaller headlines
        and headline.width > page_width * 0.10  # Lowered width requirement
        and centroid_y < page_height * 0.7  # Increased range to capture side and top headlines
    )

def safe_crop(image, coordinates):
    x_1, y_1, x_2, y_2 = map(int, coordinates)
    return image.crop((x_1, y_1, x_2, y_2))

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale for better OCR performance
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Improve binarization
    return image

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text).strip()

def quality_check(text):
    return len(text.split()) > 2  # Ensure the text is at least 3 words long

def extract_headline_from_pdf(pdf_path):
    images = pdf2image.convert_from_path(
        pdf_path,
        dpi=300
    )
    if not images:
        return "No pages found in PDF", []

    all_headlines = []
    layout_list = []
    page_numbers = []

    model = load_model()
    ocr_agent = lp.TesseractAgent(languages='eng', config='--psm 6 --oem 3')

    for page_number, image in enumerate(images, start=1):
        layout = model.detect(image)
        layout_list.append(layout)

        headlines = [b for b in layout if b.type == 'Title']
        headlines = [h for h in headlines if is_valid_headline(h.block, image.width, image.height)]
        headlines.sort(key=lambda x: (x.block.y_1, -x.block.area))

        for headline in headlines:
            x_1, y_1, x_2, y_2 = headline.coordinates
            cropped_image = safe_crop(image, (x_1, y_1, x_2, y_2))
            cropped_image = preprocess_image(cropped_image)
            headline_text = ocr_agent.detect(cropped_image).strip()
            headline_text = clean_text(headline_text)

            if quality_check(headline_text):
                all_headlines.append(f"{headline_text}")
                page_numbers.append(f"Page {page_number}")

    return all_headlines if all_headlines else ["No headlines detected"], layout_list, page_numbers

toi_image_url = "https://www.connectclue.com/uploads/eetArya1612440307545TimesofIndiaCirculationinIndiaconnectclue.jpg"
et_image_url = 'https://img.etimg.com/photo/msid-74451948,quality-100/et-logo.jpg'

if np is not None:
    if np == 'Times of India':
        response = requests.get(toi_image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Times of India Circulation", use_container_width=True)
        else:
            st.error("Failed to load the image. Please check the URL.")
    elif np == 'Economic Times':
        response = requests.get(et_image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Economic Times Circulation", use_container_width=True)
        else:
            st.error("Failed to load the image. Please check the URL.")

if np == 'Times of India':
    pdf_path = "/content/reduced_pages_toi.pdf"
    headlines, layouts, page_numbers = extract_headline_from_pdf(pdf_path)
    print("Detected Headlines:")
    print('\n'.join(headlines))
    print("\nLayout Objects:", layouts)

    df = pd.DataFrame({
        'Page Number': page_numbers,
        'Headlines': headlines
    })
elif np == 'Economic Times':
    pdf_path = "/content/reduced_pages_et.pdf"
    headlines, layouts, page_numbers = extract_headline_from_pdf(pdf_path)
    print("Detected Headlines:")
    print('\n'.join(headlines))
    print("\nLayout Objects:", layouts)

    df = pd.DataFrame({
        'Page Number': page_numbers,
        'Headlines': headlines
    })

if np is not None:
    st.write(f'Here are today\'s top headlines in {np}.')
    st.dataframe(df)
