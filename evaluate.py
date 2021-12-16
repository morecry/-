import torch
import os
from transform import MaskDataTransform
from dataset import PoemDataset
from transformers import BertForMaskedLM, BertModel, BertConfig, BertTokenizerFast
from torch.utils.data import DataLoader
from model import MaskModel


def eval(args, model=None, eval_dataloader=None, device=None):
    parallel = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        parallel = True
    if device is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if eval_dataloader is None:
        tokenizer = BertTokenizerFast.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True, clean_text=False)
        data_transform = MaskDataTransform(tokenizer=tokenizer, max_len=args.max_context_length)
        eval_dataset = PoemDataset(os.path.join(args.train_dir, 'test.txt'), transform=data_transform, sample_cnt=None)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=eval_dataset.batchify, shuffle=False)
    if model is None:
        bert_config = BertConfig.from_json_file(os.path.join(args.bert_model, 'config.json'))
        pretrained_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
        bert = BertModel.from_pretrained(args.bert_model, state_dict=torch.load(pretrained_model_file, map_location="cpu"))
        model = BertForMaskedLM(bert_config)
        model.bert = bert
        print('Loading parameters from', pretrained_model_file)
        if not parallel:
            model.load_state_dict(torch.load(pretrained_model_file, map_location="cpu"))
        else:
            model.module.load_state_dict(torch.load(pretrained_model_file, map_location="cpu"))
    model.eval()
    len_dataloader = len(eval_dataloader)
    print("\nEvaluating...", end="")
    valid_loss = 0
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs_batch, labels_batch = batch
        with torch.no_grad():
            output = model(input_ids = inputs_batch, labels = labels_batch)
            loss = output.loss
            if parallel:
                loss = loss.mean()
            valid_loss += loss
            per = (step + 1) / len_dataloader * 100
        print('\r\t\t%.3f%%' % per, end="")

    result = {'loss': valid_loss/len_dataloader}

    return result