import zipfile
import os
import sys

def extract_with_progress(zip_path, extract_dir):
    try:
        # Open with allowZip64=True for large files
        with zipfile.ZipFile(zip_path, 'r', allowZip64=True) as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            print(f"Found {total_files} files in archive")
            
            # Calculate total size
            total_size = sum(zip_ref.getinfo(name).file_size for name in file_list)
            extracted_size = 0
            
            print(f"Total size: {total_size / (1024*1024):.1f} MB")
            
            # Extract files with progress
            for i, name in enumerate(file_list, 1):
                try:
                    # Extract single file
                    zip_ref.extract(name, extract_dir)
                    
                    # Update progress
                    file_size = zip_ref.getinfo(name).file_size
                    extracted_size += file_size
                    progress = (extracted_size / total_size) * 100
                    
                    # Print progress every 100 files or for large files
                    if i % 100 == 0 or file_size > 1024*1024:  # 1MB
                        print(f"Progress: {progress:.1f}% ({i}/{total_files} files)")
                        
                except Exception as e:
                    print(f"Error extracting {name}: {e}")
                    continue
                    
            print("\nExtraction complete!")
            
    except zipfile.BadZipFile:
        print("Error: File is corrupted or not a valid ZIP file")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    zip_path = 'Reference Materials/data/RID/m1655470.zip'
    extract_dir = 'Reference Materials/data/RID/extracted'
    
    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Attempting to extract: {zip_path}")
    print(f"To directory: {extract_dir}")
    
    # Check if file exists
    if not os.path.exists(zip_path):
        print(f"Error: File not found: {zip_path}")
        sys.exit(1)
        
    # Check file size
    file_size = os.path.getsize(zip_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    # Start extraction
    extract_with_progress(zip_path, extract_dir)
