import subprocess
import os

# Create extraction directory
extract_dir = 'Reference Materials/data/RID/extracted'
os.makedirs(extract_dir, exist_ok=True)

# Paths
zip_path = 'Reference Materials/data/RID/m1655470.zip'
seven_zip_paths = [
    r'C:\Program Files\7-Zip\7z.exe',
    r'C:\Program Files (x86)\7-Zip\7z.exe'
]

def find_7zip():
    for path in seven_zip_paths:
        if os.path.exists(path):
            return path
    return None

print("Looking for 7-Zip...")
seven_zip = find_7zip()

if seven_zip:
    print(f"Found 7-Zip at: {seven_zip}")
    print("Starting extraction...")
    
    try:
        # Run 7-Zip extraction
        cmd = [seven_zip, 'x', zip_path, f'-o{extract_dir}', '-y']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        # Get any errors
        _, stderr = process.communicate()
        if stderr:
            print("Errors:", stderr)
            
        if process.returncode == 0:
            print("Extraction completed successfully!")
        else:
            print(f"Extraction failed with return code: {process.returncode}")
            
    except Exception as e:
        print(f"Error during extraction: {e}")
else:
    print("7-Zip not found. Please install 7-Zip first.")
