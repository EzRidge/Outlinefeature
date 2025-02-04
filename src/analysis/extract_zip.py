import zipfile
import os

# Create extraction directory if it doesn't exist
extract_dir = 'Reference Materials/data/RID/extracted'
os.makedirs(extract_dir, exist_ok=True)

print("Starting extraction...")
try:
    with zipfile.ZipFile('Reference Materials/data/RID/m1655470.zip', 'r') as zip_ref:
        # Get total number of files
        total_files = len(zip_ref.namelist())
        print(f"Total files to extract: {total_files}")
        
        # Extract with progress
        for i, member in enumerate(zip_ref.namelist(), 1):
            if i % 100 == 0:  # Print progress every 100 files
                print(f"Extracting file {i} of {total_files} ({(i/total_files)*100:.1f}%)")
            zip_ref.extract(member, extract_dir)
            
    print("Extraction complete!")
except Exception as e:
    print(f"Error during extraction: {e}")
