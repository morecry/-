import torch
import os
from transformers import BertModel, BertConfig, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing as mp
import torch.nn.functional as F
from model import MaskModel

class PoemDataset(Dataset):
    def __init__(self, file_path, tokenizer, sample_cnt=None):
        self.tokenizer = tokenizer
        self.data_source = []
        with open(file_path, encoding='utf-8') as f:
            self.data_source = f.read().strip().split('\n')
        if sample_cnt is not None:
            self.data_source = self.data_source[:sample_cnt]

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        return self.data_source[index]
bert_dir = 'bert'
tokenizer = BertTokenizerFast.from_pretrained(bert_dir)

def transform_data(line):
    res = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
    return res

def create_index(data_dir):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert_config = BertConfig.from_json_file(os.path.join(bert_dir, 'config.json'))
    tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
    dataset = PoemDataset(os.path.join(data_dir, 'all_pair_7.txt'), tokenizer)

    print("start create index...")
    pretrained_model_file_poem = os.path.join("output", "bert_poem.bin")
    bert_poem = BertModel(bert_config)
    checkpoint = torch.load(pretrained_model_file_poem, map_location="cpu")
    bert_poem.load_state_dict(checkpoint)
    bert_poem = bert_poem.to(device)
    bert_poem.eval()
    index = []
    batch_size = 16384
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            inputs = []
            with mp.Pool(processes=mp.cpu_count()) as pool:
                for input in pool.imap(transform_data, batch):
                    inputs.append(input)
            inputs = torch.tensor(inputs)
            inputs = inputs.to(device)

            hidden_state = bert_poem(input_ids=inputs)[0][:,0,:]
            hidden_state = hidden_state.cpu()
            index.append(hidden_state)
        index = torch.cat(index, dim=0)
        torch.save(index, os.path.join(data_dir, "index.pt"))
    print("finished!")

def query(inputs, topk=10, type=None):
    assert type in ['poem', 'oral']
    bert_dir = 'bert'
    data_dir = 'data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert_config = BertConfig.from_json_file(os.path.join(bert_dir, 'config.json'))
    tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
    dataset = PoemDataset(os.path.join(data_dir, 'all_pair_7.txt'), tokenizer)

    index = torch.load(os.path.join(data_dir, "index.pt"))
    index = index.to(device)
    if type == 'oral':
        pretrained_model_file_ch = os.path.join("output", "bert_ch.bin")
    elif type == 'poem':
        pretrained_model_file_ch = os.path.join("output", "bert_poem.bin")
    bert = BertModel(bert_config)
    checkpoint = torch.load(pretrained_model_file_ch, map_location="cpu")
    bert.load_state_dict(checkpoint)
    bert = bert.to(device)
    bert.eval()
    inputs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs))

    inputs = torch.tensor([inputs])
    inputs = inputs.to(device)
    with torch.no_grad():
        hidden_state = bert(input_ids=inputs)[0][:, 0, :]
        sim = F.cosine_similarity(hidden_state, index)
    values, indices = sim.topk(topk, sorted=True)
    results = [dataset[idx] for idx in indices]
    return results, values

def predict(input):

    bert_dir = 'bert'
    data_dir = 'data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert_config = BertConfig.from_json_file(os.path.join(bert_dir, 'config.json'))
    tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
    dataset = PoemDataset(os.path.join(data_dir, 'all_pair.txt'), tokenizer)

    pretrained_model_file_ch = os.path.join("output", "PoemBert_mask.bin")
    bert = BertModel(bert_config)
    checkpoint = torch.load(pretrained_model_file_ch, map_location="cpu")
    bert.load_state_dict(checkpoint)
    model = MaskModel(bert_config, bert)
    model = model.to(device)
    model.eval()
    inputs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    spec_token_id = tokenizer.convert_tokens_to_ids('#')
    mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    mask_index = []
    for i, token_id in enumerate(inputs):
        if token_id == spec_token_id:
            mask_index.append(i)

    inputs = torch.tensor([inputs])
    inputs = inputs.to(device)
    inputs[0, mask_index] =mask_token_id
    print(inputs)
    with torch.no_grad():
        hidden_state = model(inputs)[0, mask_index, :]
    print(hidden_state.shape)
    values, indices = hidden_state.topk(1)
    print(values, indices)
    indices = indices.squeeze().cpu().tolist()
    print(indices)
    char = tokenizer.convert_ids_to_tokens(indices)
    print(char)
    breakpoint()
    # values, indices = sim.topk(topk, sorted=True)
    # results = [dataset[idx] for idx in indices]
    return results, values

if  __name__ == "__main__":
    data_dir = 'data'
    if not os.path.exists(os.path.join(data_dir, "index.pt")):
        create_index(data_dir)
    # input = "长风破浪会有时，直挂云帆济沧海。"
    input = "烟笼寒水月笼沙，夜泊秦淮近酒家。"
    results, values = query(input, topk=20, type='poem') #type为poem表示输入为诗歌，oral表示输入为日常用语

    # input = "夜晚睡不着"
    # input = "冬天早上天气好冷啊"
    # input = "和朋友欢聚，饮酒"
    # results, values = query(input, topk=20, type='oral')
    print("\n#输入: " + input)
    print("=" * 50)
    print(" " * 8 + "#检索结果(Top%d)"%len(results))
    for poem in results:
        print(poem)
    # print(predict("昏#病眼已经年，世味#酸总澹然。"))






