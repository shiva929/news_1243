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
from scraper import get_google_drive_links, get_direct_download_link, download_pdf
from pdf_utils import extract_pages

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_colwidth', None)
st.title("The Digital Times")

# Initialize LayoutParser model
@lru_cache(maxsize=1)
def load_model():
    return lp.models.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]  # Higher confidence threshold
    )

def is_valid_headline(headline, page_width, page_height):
    # More robust headline validation
    return (
        headline.height > page_height * 0.02 and  # Minimum height threshold
        headline.width > page_width * 0.15 and    # Minimum width threshold
        headline.y_1 < page_height * 0.3         # Must be in top 30% of page
    )

def safe_crop(image, coordinates):
    x_1, y_1, x_2, y_2 = map(int, coordinates)
    # Add padding to prevent cropping errors
    padding = 5
    x_1 = max(0, x_1 - padding)
    y_1 = max(0, y_1 - padding)
    x_2 = min(image.width, x_2 + padding)
    y_2 = min(image.height, y_2 + padding)
    return image.crop((x_1, y_1, x_2, y_2))

def preprocess_image(image):
    # Enhanced preprocessing
    image = image.convert("L")  # Grayscale
    return image

def clean_text(text):
    # Improved text cleaning
    text = re.sub(r'[^a-zA-Z0-9 \-\'\"]', '', text)
    return ' '.join(text.split()).strip()

def extract_headline_from_pdf(pdf_path):
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=200)  # Adjusted DPI
        if not images:
            return ["No pages found in PDF"], [], []

        model = load_model()
        ocr_agent = lp.TesseractAgent(languages='eng')
        
        all_headlines = []
        page_numbers = []
        
        for page_num, image in enumerate(images, 1):
            layout = model.detect(image)
            headlines = [block for block in layout if block.type == 'Title']
            
            for headline in headlines:
                if is_valid_headline(headline.block, image.width, image.height):
                    cropped_img = safe_crop(image, headline.coordinates)
                    cropped_img = preprocess_image(cropped_img)
                    
                    text = ocr_agent.detect(cropped_img)
                    clean_text = clean_text(text)
                    
                    if len(clean_text.split()) >= 3:  # Minimum 3 words
                        all_headlines.append(clean_text)
                        page_numbers.append(f"Page {page_num}")

        return all_headlines or ["No headlines detected"], [], page_numbers
        
    except Exception as e:
        return [f"Error processing PDF: {str(e)}"], [], []

# URLs and UI setup
toi_url = "https://www.dailyepaper.in/times-of-india-epaper-pdf-march-2025/"
et_url = 'https://www.dailyepaper.in/economic-times-newspaper-2025/'
toi_image_url = "https://www.connectclue.com/uploads/eetArya1612440307545TimesofIndiaCirculationinIndiaconnectclue.jpg"
et_image_url = "https://img.etimg.com/photo/msid-74451948,quality-100/et-logo.jpg"

# Sidebar Selection
np = st.sidebar.selectbox("Choose Newspaper", ["Times of India", "Economic Times"])

if np:
    # Display Image
    image_url = toi_image_url if np == 'Times of India' else et_image_url
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption=f"{np} Logo", use_column_width=True)
    except:
        st.warning("Couldn't load newspaper image")

    # Download and process PDF
    if st.button(f"Get Today's {np} Headlines"):
        with st.spinner(f"Downloading {np}..."):
            try:
                links = get_google_drive_links(toi_url if np == 'Times of India' else et_url)
                if not links:
                    st.error("Couldn't fetch PDF links")
                    return
                    
                direct_link = get_direct_download_link(links[1])
                raw_pdf = f"{np.replace(' ', '_').lower()}_raw.pdf"
                reduced_pdf = f"{np.replace(' ', '_').lower()}_reduced.pdf"
                
                if download_pdf(direct_link, raw_pdf):
                    extract_pages(raw_pdf, [2, 3, 4], reduced_pdf)  # Sample pages
                    
                    with st.spinner("Analyzing headlines..."):
                        headlines, _, pages = extract_headline_from_pdf(reduced_pdf)
                        
                        if headlines and headlines[0] != "No headlines detected":
                            st.success(f"Top Headlines from {np}:")
                            df = pd.DataFrame({
                                'Page': pages,
                                'Headline': headlines
                            })
                            st.dataframe(df)
                        else:
                            st.warning("No headlines found in the selected pages")
            except Exception as e:
                st.error(f"Error processing newspaper: {str(e)}")
