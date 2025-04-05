import requests
from bs4 import BeautifulSoup

def get_google_drive_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True) if "drive.google.com" in a["href"]]

def get_direct_download_link(google_drive_link):
    file_id = google_drive_link.split("/d/")[1].split("/")[0]
    return f"https://drive.google.com/uc?id={file_id}&export=download"

def download_pdf(direct_link, filename):
    response = requests.get(direct_link)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully!")
        return True
    else:
        print(f"Failed to download {filename}")
        return False
