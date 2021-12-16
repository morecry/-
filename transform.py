import torch
import random


class MaskDataTransform(object):
    def __init__(self, tokenizer, word = 7):
        self.tokenizer = tokenizer
        self.word = word
        self.max_len = word*2 + 2 + 2 #开始 结尾 逗号 句号
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.pad_id = 0

    def __call__(self, sentence):
        inputs = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
        mask_idx_1 = random.randint(1, self.word)
        mask_idx_2 = random.randint(self.word + 2, 2 * self.word + 1)
        label = [-100] * len(inputs)
        label[mask_idx_1] = inputs[mask_idx_1]
        label[mask_idx_2] = inputs[mask_idx_2]
        inputs[mask_idx_1] = self.mask_id
        inputs[mask_idx_2] = self.mask_id
        return inputs, label

class NxtDataTransform(object):
    def __init__(self, tokenizer, word = 7):
        self.tokenizer = tokenizer
        self.word = word
        self.max_len = word*2 + 2 + 2 #开始 结尾 逗号 句号
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad_id = 0

    def __call__(self, pre, pst, neg):
        pre = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pre))[1:-1]
        pst = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pst))[1:-1]
        neg = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(neg))[1:-1]
        cat_1 = [self.cls_id] + pre + [self.sep_id] + pst + [self.sep_id]
        cat_2 = [self.cls_id] + pre + [self.sep_id] + neg + [self.sep_id]
        cat_3 = [self.cls_id] + neg + [self.sep_id] + pst + [self.sep_id]
        res_1 = [cat_1, cat_2]
        res_2 = [cat_1, cat_3]
        return res_1, res_2

class MatchDataTransform(object):
    def __init__(self, tokenizer, word = 7):
        self.tokenizer = tokenizer
        self.word = word
        self.max_len = word*2 + 2 + 2 #开始 结尾 逗号 句号
        self.ch_maxlen =25
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad_id = 0

    def __call__(self, ch, poems):
        ch = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ch))[1:-1]
        if len(ch) >= self.ch_maxlen:
            ch = ch[:self.ch_maxlen]
            ch_mask = [1] * self.ch_maxlen
        else:
            ch = ch + [self.pad_id]*(self.ch_maxlen - len(ch))
            ch_mask = [1] * len(ch) + [0]*(self.ch_maxlen - len(ch))
        ch = [self.cls_id] + ch + [self.sep_id]
        ch_mask = [1] + ch_mask + [1]
        poem_inputs = []
        for poem in poems:
            poem = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(poem))
            poem_inputs.append(poem)
        return ch, ch_mask, poem_inputs


