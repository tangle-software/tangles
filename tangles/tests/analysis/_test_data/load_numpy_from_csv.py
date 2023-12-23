import numpy as np
import os

def load(filename:str, dtype=int):
    lines = []
    
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, filename)
    with open(abs_file_path, 'r') as f:
        line = f.readline()
        while line is not None and len(line) > 0:
            if line[0] != '#':
                lines.append(line.split(' '))
            line = f.readline()
    return np.array(lines).astype(dtype)