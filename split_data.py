import os
import random
file_path = os.path.join('data', 'all_pair_7.txt')
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.read().strip().split('\n')
random.shuffle(lines)
valid_rate = 0.01
valid_set = lines[:int(len(lines)*0.01)]
train_set = lines[int(len(lines)*0.01):]
train_path = os.path.join('data', 'train.txt')
valid_path = os.path.join('data', 'valid.txt')
with open(train_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_set))
with open(valid_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_set))
