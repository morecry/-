import os
import time
import logging
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertModel, BertConfig, BertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from dataset import MaskDataset, NxtDataset, MatchDataset
from transform import MaskDataTransform, NxtDataTransform, MatchDataTransform
from model import MaskModel, NxtModel, MatchModel

logging.basicConfig(level=logging.ERROR)

def eval(args, model=None, eval_dataloader=None, device=None, parallel=None, type=None):
    assert type in ['mask', 'nxt', 'match']
    model.eval()
    len_dataloader = len(eval_dataloader)
    print("\nEvaluating...", end="")
    valid_loss = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            if type == 'mask':
                batch = tuple(t.to(device) for t in batch)
                inputs_batch, labels_batch = batch
                loss = model(inputs_batch, labels_batch)
            elif type == 'nxt':
                batch = batch.to(device)
                loss = model(batch)
            elif type == 'match':
                batch = tuple(t.to(device) for t in batch)
                ch_batch, ch_mask_batch, poem_batch = batch
                loss = model(ch_batch, ch_mask_batch, poem_batch)
            if parallel:
                loss = loss.mean()
            valid_loss += loss
            per = (step + 1) / len_dataloader * 100
        print('\r\t\t%.3f%%' % per, end="")

    result = {'loss': valid_loss/len_dataloader}
    return result

def train(args, type=None):
    assert type in ['mask', 'nxt', 'match']
    parallel = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        parallel = True
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print_freq = args.print_freq

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(args.bert_model))
    bert_config = BertConfig.from_json_file(os.path.join(args.bert_model, 'config.json'))

    if type == 'mask':
        state_save_path = os.path.join(args.output_dir, 'PoemBert_mask.bin')
        data_source_file_train = os.path.join(args.data_dir, 'data_source_mask_train.bin')
        data_source_file_eval = os.path.join(args.data_dir, 'data_source_mask_eval.bin')
        log_wf = open(os.path.join(args.output_dir, 'log_mask.txt'), 'a', encoding='utf-8')
        mask_transform = MaskDataTransform(tokenizer=tokenizer, word=7)
        train_dataset = MaskDataset(os.path.join(args.data_dir, 'train.txt'), data_source_file=data_source_file_train, transform=mask_transform, word=7,
                                    sample_cnt=None)
        eval_dataset = MaskDataset(os.path.join(args.data_dir, 'valid.txt'), data_source_file=data_source_file_eval, transform=mask_transform, word=7,
                                   sample_cnt=None)
        pretrained_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
        print('Loading parameters from', pretrained_model_file)
        log_wf.write('Loading parameters from %s' % pretrained_model_file + '\n')
        bert = BertModel.from_pretrained(args.bert_model,
                                         state_dict=torch.load(pretrained_model_file, map_location="cpu"))
        model = MaskModel(bert_config, bert)
    elif type == 'nxt':
        state_save_path = os.path.join(args.output_dir, 'PoemBert_nxt.bin')
        data_source_file_train = os.path.join(args.data_dir, 'data_source_nxt_train.bin')
        data_source_file_eval = os.path.join(args.data_dir, 'data_source_nxt_eval.bin')
        log_wf = open(os.path.join(args.output_dir, 'log_nxt.txt'), 'a', encoding='utf-8')
        nxt_transform = NxtDataTransform(tokenizer=tokenizer, word=7)
        train_dataset = NxtDataset(os.path.join(args.data_dir, 'train.txt'), data_source_file=data_source_file_train, transform=nxt_transform, word=7,
                                   sample_cnt=None)
        eval_dataset = NxtDataset(os.path.join(args.data_dir, 'valid.txt'), data_source_file=data_source_file_eval,transform=nxt_transform, word=7,
                                  sample_cnt=None)
        pretrained_model_file = os.path.join(args.output_dir, 'PoemBert_mask.bin')
        print('Loading parameters from', pretrained_model_file)
        checkpoint = torch.load(pretrained_model_file, map_location="cpu")
        bert = BertModel(bert_config)
        bert.load_state_dict(checkpoint)
        model = NxtModel(bert_config, bert)

    elif type == 'match':
        state_save_path_poem = os.path.join(args.output_dir, 'bert_poem.bin')
        state_save_path_ch = os.path.join(args.output_dir, 'bert_ch.bin')
        data_source_file_train = os.path.join(args.data_dir, 'data_source_match_train.bin')
        data_source_file_eval = os.path.join(args.data_dir, 'data_source_match_eval.bin')
        log_wf = open(os.path.join(args.output_dir, 'log_match.txt'), 'a', encoding='utf-8')
        match_transform = MatchDataTransform(tokenizer=tokenizer, word=7)
        train_dataset = MatchDataset(os.path.join(args.data_dir, 'train_match_7.txt'), data_source_file=data_source_file_train, transform=match_transform, word=7,
                                     sample_cnt=None)
        eval_dataset = MatchDataset(os.path.join(args.data_dir, 'valid_match_7.txt'), data_source_file=data_source_file_eval, transform=match_transform, word=7,
                                    sample_cnt=None)
        poem_pretrained_model_file = os.path.join(args.output_dir, 'PoemBert_nxt.bin')
        ch_pretrained_model_file = os.path.join(args.bert_model, 'pytorch_model.bin')
        print('Loading parameters from', poem_pretrained_model_file, ch_pretrained_model_file)
        checkpoint_poem = torch.load(poem_pretrained_model_file, map_location="cpu")
        bert_poem = BertModel(bert_config)
        bert_poem.load_state_dict(checkpoint_poem)
        bert_ch = BertModel.from_pretrained(args.bert_model,
                                            state_dict=torch.load(ch_pretrained_model_file, map_location="cpu"))
        model = MatchModel(bert_config, bert_poem, bert_ch)

    print("train set num: %d" % len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.batchify,
                                  shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=eval_dataset.batchify,
                                 shuffle=False, num_workers=8)

    if parallel:
        model = nn.DataParallel(model)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    best_eval_result = 1e8
    if type == 'match':
        print_freq = 1
    for epoch in range(1, int(args.num_train_epochs) + 1):
        train_loss = 0
        with tqdm(total=len(train_dataloader)) as bar:
            for step, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()
                if type == 'mask':
                    batch = tuple(t.to(device) for t in batch)
                    inputs_batch, labels_batch = batch
                    loss = model(inputs_batch, labels_batch)
                elif type == 'nxt':
                    batch = batch.to(device)
                    loss = model(batch)
                elif type == 'match':
                    batch = tuple(t.to(device) for t in batch)
                    ch_batch, ch_mask_batch, poem_batch = batch
                    loss = model(ch_batch, ch_mask_batch, poem_batch)

                if parallel:
                    loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                train_loss += loss.item()

                if (step + 1) % print_freq == 0:
                    bar.update(print_freq)
                    output_string = "\nepoch: %d\t|step: %d\t| loss: %.8f" % (epoch, step + 1, train_loss/(step + 1))
                    print(output_string)
                    log_wf.write(output_string)
                log_wf.flush()

        # --------------------------Eval after Every Epoch----------------------------------------------
        eval_result = eval(args, model, eval_dataloader, device, parallel, type)
        output_string = "\n" + "EVAL RESULT     Epoch %d      Step %d" % (epoch, step + 1) + "\n" + "="*50 + "\n" + \
                        "loss:%.8f" % eval_result['loss'] + "\t###" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n'+ "="*50 + '\n'
        print(output_string)
        log_wf.write(output_string)
        if type != 'match':
            print('[Saving at]', state_save_path)
            log_wf.write('[Saving at] %s\n' % state_save_path)
        else:
            print('[Saving at]', state_save_path_ch, state_save_path_poem)
            log_wf.write('[Saving at] %s %s\n' % (state_save_path_ch, state_save_path_poem))
        if not parallel:
            if type != 'match':
                torch.save(model.bert.state_dict(), state_save_path)
            else:
                torch.save(model.bert_ch.state_dict(), state_save_path_ch)
                torch.save(model.bert_poem.state_dict(), state_save_path_poem)
        else:
            if type != 'match':
                torch.save(model.module.bert.state_dict(), state_save_path)
            else:
                torch.save(model.module.bert_ch.state_dict(), state_save_path_ch)
                torch.save(model.module.bert_poem.state_dict(), state_save_path_poem)
        log_wf.flush()