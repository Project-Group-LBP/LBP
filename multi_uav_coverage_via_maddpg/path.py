import os
from pathlib import Path

def generate_tree(directory, prefix="", is_last=True):
    """Generate a tree-like directory structure diagram."""
    output = []
    entries = sorted(list(os.scandir(directory)), key=lambda e: e.name)
    
    for i, entry in enumerate(entries):
        is_last_entry = i == len(entries) - 1
        marker = "└── " if is_last_entry else "├── "
        
        output.append(f"{prefix}{marker}{entry.name}")
        
        if entry.is_dir():
            extension = "    " if is_last_entry else "│   "
            output.extend(generate_tree(entry.path, prefix + extension, is_last_entry))
    
    return output

def print_tree(directory):
    """Print the directory tree starting from the given directory."""
    print(directory)
    tree = generate_tree(directory)
    print("\n".join(tree))

# Use it on your current directory
if __name__ == "__main__":
    current_dir = "."  # Or specify your path
    print_tree(current_dir)