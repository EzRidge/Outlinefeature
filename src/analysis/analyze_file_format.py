def print_hex_blocks(filename, block_size=16, num_blocks=4):
    print(f"Analyzing file: {filename}")
    print(f"Reading first {num_blocks} blocks of {block_size} bytes each\n")
    
    with open(filename, 'rb') as f:
        for block_num in range(num_blocks):
            # Read block
            block = f.read(block_size)
            if not block:
                break
                
            # Print offset
            print(f"Offset {block_num * block_size:04x}:")
            
            # Print hex values
            hex_values = ' '.join(f'{b:02x}' for b in block)
            print(f"Hex: {hex_values}")
            
            # Print ASCII representation
            ascii_values = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in block)
            print(f"ASCII: {ascii_values}\n")

if __name__ == "__main__":
    filename = 'Reference Materials/data/RID/m1655470.zip'
    print_hex_blocks(filename)
