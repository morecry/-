import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForPreTraining

class MaskModel(BertForPreTraining):
    def __init__(self, config, bert):
        super(MaskModel, self).__init__(config)
        self.bert = bert
        self.vocab_size = config.vocab_size
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, inputs, labels=None):
        output_hidden_states = self.bert(input_ids=inputs)[0]
        output_token_prob = self.linear(output_hidden_states)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output_token_prob.view(-1, self.vocab_size), labels.view(-1))
            return loss
        else:
            return output_token_prob

class NxtModel(BertForPreTraining):
    def __init__(self, config, bert):
        super(NxtModel, self).__init__(config)
        self.bert = bert
        self.vocab_size = config.vocab_size
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, inputs):
        batch, neg, length = inputs.shape
        inputs = inputs.view(-1, length)
        output_hidden_states = self.bert(input_ids=inputs)[0][:,0,:]
        # print(output_hidden_states.shape)
        output_score = self.linear(output_hidden_states)
        output_score = output_score.view(-1, neg)
        loss = -F.log_softmax(output_score, -1)[:, 0].mean()
        return loss

class MatchModel(BertForPreTraining):
    def __init__(self, config, bert_poem, bert_ch):
        super(MatchModel, self).__init__(config)
        self.bert_poem = bert_poem
        self.bert_ch = bert_ch
        self.vocab_size = config.vocab_size

    def forward(self, ch_batch, ch_mask_batch, poem_batch):

        batch, neg, length = poem_batch.shape
        poem_batch = poem_batch.view(-1, length)
        poem_output_hidden_states = self.bert_poem(input_ids=poem_batch)[0][:, 0, :]
        poem_output_hidden_states = poem_output_hidden_states.view(batch, neg, -1) #shape: batch * neg * hidden_size
        ch_output_hidden_states = self.bert_ch(input_ids=ch_batch, attention_mask=ch_mask_batch)[0][:, 0, :] #shape: batch * hidden_size
        ch_output_hidden_states = ch_output_hidden_states.unsqueeze(1) #shape: batch * 1 * hidden_size
        match_score = torch.bmm(ch_output_hidden_states, poem_output_hidden_states.permute(0,2,1)).squeeze(1) #shape:batch * neg
        loss = -F.log_softmax(match_score, -1)[:, 0].mean()
        return loss
