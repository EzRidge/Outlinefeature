import zipfile
import os
import sys
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA

def try_open_with_compression(zip_path):
    compression_methods = [
        (ZIP_STORED, "Stored (no compression)"),
        (ZIP_DEFLATED, "Deflate"),
        (ZIP_BZIP2, "BZIP2"),
        (ZIP_LZMA, "LZMA")
    ]
    
    for method, name in compression_methods:
        try:
            print(f"\nTrying {name} compression method...")
            with ZipFile(zip_path, 'r', compression=method, allowZip64=True) as zf:
                # Try to read file info
                file_list = zf.namelist()
                print(f"Success! Found {len(file_list)} files")
                print("First few files:")
                for f in file_list[:5]:
                    print(f"- {f}")
                return zf.compression, file_list
        except Exception as e:
            print(f"Failed with {name}: {str(e)}")
    
    return None, None

def extract_files(zip_path, extract_dir, compression):
    try:
        with ZipFile(zip_path, 'r', compression=compression, allowZip64=True) as zf:
            print("\nStarting extraction...")
            total_files = len(zf.namelist())
            for i, name in enumerate(zf.namelist(), 1):
                try:
                    print(f"Extracting {i}/{total_files}: {name}")
                    zf.extract(name, extract_dir)
                except Exception as e:
                    print(f"Error extracting {name}: {e}")
            print("\nExtraction complete!")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    zip_path = 'Reference Materials/data/RID/m1655470.zip'
    extract_dir = 'Reference Materials/data/RID/extracted'
    
    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Analyzing: {zip_path}")
    compression, file_list = try_open_with_compression(zip_path)
    
    if compression is not None:
        print("\nFound valid ZIP format!")
        response = input("Would you like to proceed with extraction? (y/n): ")
        if response.lower() == 'y':
            extract_files(zip_path, extract_dir, compression)
    else:
        print("\nCould not open file with any compression method")
