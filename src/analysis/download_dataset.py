import requests
import os
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file with progress bar
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Send a HEAD request first to get the file size
    response = requests.head(url, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    print(f"\nDownloading {filename}")
    print(f"Expected size: {total_size / (1024*1024):.2f} MB")
    
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    # Verify download
    actual_size = os.path.getsize(filename)
    print(f"\nDownload complete!")
    print(f"Actual size: {actual_size / (1024*1024):.2f} MB")
    
    if actual_size != total_size:
        print("WARNING: Downloaded file size doesn't match expected size!")
        return False
    return True

if __name__ == "__main__":
    # URL from the DOI page
    url = "https://dataserv.ub.tum.de/s/m1655470/download"
    filename = "Reference Materials/data/RID/m1655470_new.zip"
    
    print("Starting download of RID dataset...")
    success = download_file(url, filename)
    
    if success:
        print("\nDownload successful! New file is ready at:")
        print(filename)
    else:
        print("\nDownload may be incomplete. Please verify the file.")
