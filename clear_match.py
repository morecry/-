import os
import multiprocessing as mp
from tqdm import tqdm

def process(line):
    poem = line.split('\t')[1]
    if len(poem) == extract_type:
        return line
    else:
        return None

extract_type = 7

for mode in ['train', 'valid']:
    data_file = os.path.join("data", "%s_match.txt" % mode)
    out_file = os.path.join("data", "%s_match_%d.txt" % (mode, extract_type))
    with open(data_file, 'r') as f:
        all_lines = f.read().strip().split('\n')
    result = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for e in pool.imap(process, tqdm(all_lines, total=len(all_lines))):
            if e is not None:
                result.append(e)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))