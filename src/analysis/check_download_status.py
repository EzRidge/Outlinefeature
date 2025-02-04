import os
import hashlib

def get_file_info(filepath):
    print(f"Checking file: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print("Error: File not found")
        return
    
    # Get file size
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    
    # Calculate MD5 hash of first 1MB to check for corruption
    print("\nCalculating hash of first 1MB...")
    with open(filepath, 'rb') as f:
        first_mb = f.read(1024 * 1024)
        md5_hash = hashlib.md5(first_mb).hexdigest()
        print(f"MD5 hash of first 1MB: {md5_hash}")
    
    # Check file header
    print("\nChecking file header...")
    with open(filepath, 'rb') as f:
        header = f.read(4)
        header_hex = ' '.join(f'{b:02x}' for b in header)
        print(f"First 4 bytes (hex): {header_hex}")
        
        # Common file signatures
        signatures = {
            b'PK\x03\x04': 'ZIP archive',
            b'PK\x05\x06': 'ZIP archive (empty)',
            b'PK\x07\x08': 'ZIP archive (spanned)',
            b'\x1f\x8b\x08': 'GZIP archive',
            b'BZh': 'BZIP2 archive',
            b'\x37\x7A\xBC\xAF': '7Z archive',
            b'Rar!\x1A\x07': 'RAR archive'
        }
        
        # Try to identify file type
        file_type = "Unknown"
        for sig, desc in signatures.items():
            if header.startswith(sig):
                file_type = desc
                break
                
        print(f"File type based on signature: {file_type}")
    
    # Expected size from website
    expected_size = 1.5 * 1024 * 1024 * 1024  # 1.5 GB
    if abs(size_bytes - expected_size) > 1024 * 1024:  # Allow 1MB difference
        print("\nWARNING: File size differs significantly from expected size!")
        print(f"Expected: {expected_size / (1024*1024):.2f} MB")
        print(f"Actual: {size_mb:.2f} MB")
        print("The file might be incomplete or corrupted")

if __name__ == "__main__":
    filepath = 'Reference Materials/data/RID/m1655470.zip'
    get_file_info(filepath)
