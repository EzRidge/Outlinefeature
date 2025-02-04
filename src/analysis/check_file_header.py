def print_hex(data):
    # Print first 16 bytes in hex
    hex_bytes = ' '.join(f'{b:02x}' for b in data)
    print(f"First 16 bytes (hex): {hex_bytes}")
    
    # Try to decode as ASCII/UTF-8 for readable characters
    try:
        ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
        print(f"ASCII representation: {ascii_str}")
    except:
        print("Could not decode as ASCII")

print("Reading file header...")
with open('Reference Materials/data/RID/m1655470.zip', 'rb') as f:
    header = f.read(16)
    print_hex(header)
