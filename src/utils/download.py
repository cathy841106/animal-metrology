import os
import requests
from zipfile import ZipFile
from tqdm import tqdm

def download_file(url, dest_path):
    """
    Download a file from a URL with a progress bar.
    
    Parameters:
        url: file download URL
        dest_path: local file path to save the downloaded file
        
    Returns:
        None
    """
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def unzip_file(zip_path, extract_to, members=None):
    """
    Extract a ZIP file to a specified directory.
    
    Parameters:
        zip_path: path to the ZIP file
        extract_to: directory to extract files into
        members: optional list of specific files/folders to extract
        
    Returns:
        None
    """
    
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_to, members=members)