import torch
from torch import nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertLayer

from transformers import BertTokenizer
from cedr.modeling import BertRanker
from cedr import modeling_util

from pyterrier_bert.pyt_cedr import CEDRPipeline

import numpy as np
    
class ParadeRanker(BertRanker):
    def __init__(self, aggregation_method='transformer'):
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").cuda()
        
        self.transformer_layer_1 = BertLayer(self.bert.config).cuda()
        self.transformer_layer_2 = BertLayer(self.bert.config).cuda()
        self.num_passages = 4
        self.maxseqlen = 0
        self.linear = nn.Linear(self.bert.config.hidden_size, 1).cuda()
        
        if aggregation_method == "maxp":
            self.aggregation = self.aggregate_using_maxp
        elif aggregation_method == "transformer":
            self.aggregation = self.aggregate_using_transformer
            input_embeddings = self.bert.get_input_embeddings()
            cls_token_id = torch.tensor([[101]]).cuda()
            self.initial_cls_embedding = input_embeddings(cls_token_id).view(1, self.bert.config.hidden_size)
            #self.full_position_embeddings = torch.zeros(
            #    (1, self.num_passages + 1, self.bert.config.hidden_size), requires_grad=True, dtype=torch.float
            #).cuda()
            #torch.nn.init.normal_(self.full_position_embeddings, mean=0.0, std=0.02)
            
            # AIAYN embeddings
            def get_position_angle_vec(position, d_hid):
                return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

            sinusoid_table = np.array([get_position_angle_vec(pos_i, self.bert.config.hidden_size) for pos_i in range(100)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
            

            self.initial_cls_embedding = nn.Parameter(self.initial_cls_embedding, requires_grad=True)
            self.full_position_embeddings = nn.Parameter(torch.FloatTensor(sinusoid_table).unsqueeze(0))
        elif aggregation_method == 'average':
            self.aggregation = self.aggregate_using_avg
        else:
            raise NotImplementedError()
        
    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        batch_size = query_tok.shape[0]
        scores = None
        
        for i in range(batch_size):
            score = self.encode_bert(
                query_tok[i].unsqueeze(0),
                query_mask[i].unsqueeze(0),
                doc_tok[i].unsqueeze(0),
                doc_mask[i].unsqueeze(0)
            )

            if scores == None:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=0)
        
        return scores
        
    def tokenize(self, text):
        toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return toks
    
    def aggregate_using_avg(self, cls):
        return torch.mean(cls, 0, keepdims=True)
    
    def aggregate_using_maxp(self, cls):
        return torch.max(cls, 0, keepdims=True)[0]
    
    def aggregate_using_transformer(self, cls):
        expanded_cls = cls.view(-1, self.num_passages, self.bert.config.hidden_size)
        # TODO make sure batch size here is correct
        batch_size = expanded_cls.shape[0]
        tiled_initial_cls = self.initial_cls_embedding.repeat(batch_size, 1)
        merged_cls = torch.cat((tiled_initial_cls.view(batch_size, 1, self.bert.config.hidden_size), expanded_cls), dim=1)
        merged_cls = merged_cls + self.full_position_embeddings[:, :merged_cls.shape[1]]

        (transformer_out_1,) = self.transformer_layer_1(merged_cls, None, None, None)
        (transformer_out_2,) = self.transformer_layer_2(transformer_out_1, None, None, None)

        aggregated = transformer_out_2[:, 0, :]
        
        return aggregated
    
    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN) # MORE SENSIBLE PASSAGING
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        self.num_passages = sbcount
        
        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)
        
        cls = self.bert(toks, mask, segment_ids.long())[0][:,0,:]
        
        aggregated = self.aggregation(cls)
        
        
        return self.linear(aggregated)