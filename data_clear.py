import os
import multiprocessing as mp
from transformers import BertTokenizerFast
from tqdm import tqdm
extract_type = 7

def check(line):
    if len(line) != 2*(extract_type + 1):
        return None
    tokenizer = BertTokenizerFast.from_pretrained('bert')
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))[1:-1]
    sep_ids = [tokenizer.convert_tokens_to_ids('，'), tokenizer.convert_tokens_to_ids('。'), tokenizer.convert_tokens_to_ids('？')]
    if len(tokens) != 2*(extract_type + 1):
        return None
    start = 670
    end = 7991
    for i, token in enumerate(tokens):
        if i == extract_type or i == (2 * extract_type + 1):
            if not (token in sep_ids):
                return None
        else:
            if not (start <= token <= end):
                return None
    return line




if __name__ == '__main__':
    data_file = os.path.join("data", "all_pair.txt")
    with open(data_file, 'r') as f:
        all_lines = f.read().strip().split('\n')
    result = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for e in pool.imap(check, tqdm(all_lines, total=len(all_lines))):
            if e is not None:
                result.append(e)
    with open(os.path.join('data', 'all_pair_%d.txt' % extract_type), 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
