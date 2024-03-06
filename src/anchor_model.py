import pdb
import math
import torch
import numpy as np
from torch import nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel, BertTokenizer
import copy
       
from .attention import MultiheadAttention

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SSNTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        # self attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout, vdim=d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # cross attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, n_heads, dropout=dropout, vdim=d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src, pos, mask, query_sine_embed=None, is_first=False):
        # self attention
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)
        # content plut position
        q = q_content + q_pos
        k = k_content + k_pos
        
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(src)
        v = self.ca_v_proj(src)
        
        bsz, n_q, n_model = q_content.shape
        _, seq_len, _ = k_content.shape
        
        k_pos = self.ca_kpos_proj(pos)
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
        
        q = q.view(bsz, n_q, self.n_heads, n_model//self.n_heads)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(bsz, n_q, self.n_heads, n_model//self.n_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(bsz, n_q, n_model * 2)
        k = k.view(bsz, seq_len, self.n_heads, n_model//self.n_heads)
        k_pos = k_pos.view(bsz, seq_len, self.n_heads, n_model//self.n_heads)
        k = torch.cat([k, k_pos], dim=3).view(bsz, seq_len, n_model * 2)

        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), key_padding_mask=~mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class SSNTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, query_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = decoder_layer.d_model
        self.query_dim = query_dim
        self.query_scale = MLP(self.d_model, self.d_model, self.d_model, 2)
        self.ref_point_head = MLP(query_dim // 2 * self.d_model, self.d_model, self.d_model, 2)
        self.ref_anchor_head = MLP(self.d_model, self.d_model, 2, 2)
        self.bbox_embed = MLP(self.d_model, self.d_model, query_dim, 3)

    def forward(self, query_objects, src, pos, mask, refpoints_unsigmoid=None):
        # [bsz, n_q, 4]
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        
        output = query_objects
        for lid, layer in enumerate(self.layers):
            # A(x), [bsz, n_q, 4]
            obj_center = reference_points[..., :self.query_dim]   
            # get sine embedding for the query vector, PE(x), [bsz, n_q, 2*d_model]
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model//2)
            # Pq, [bsz, n_q, d_model]
            query_pos = self.ref_point_head(query_sine_embed)
            # transformation, MLP(Cq), [bsz, n_q, d_model]
            pos_transformation = self.query_scale(query_objects)
            # apply transformation, PE(xq,yq)*MLP(Cq), [bsz. n_q, d_model]
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation
            
            # 基于当前层的 output 生成 x, y 坐标的调制参数(向量)，对应于 paper 公式中的 
            # w_{q,ref} & h_{q,ref}, (bsz, n_q, 2)
            refHW_cond = self.ref_anchor_head(output).sigmoid()
            # 分别调制 x, y 坐标并处以 w, h 归一化，从而将尺度信息注入到交叉注意力中
            # 后 self.d_model // 2 个维度对应 x 坐标，前 self.d_model // 2 个维度对应 y 坐标；
            # query_sine_embed[..., self.d_model // 2:] 对应 paper 公式的 PE(x)
            # query_sine_embed[..., :self.d_model // 2] 对应 paper 公式的 PE(y)
            # obj_center[..., 2] 是宽，对应 paper 公式的 w_q，obj_center[..., 3] 是高，对应 paper 公式的 h_q
            query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
            query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            # 解码
            output = layer(output, query_pos, src, pos, mask, query_sine_embed=query_sine_embed, is_first=(lid == 0))
            
            # update
            tmp = self.bbox_embed(output)
            tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
            new_reference_points = tmp[..., :self.query_dim].sigmoid()
            reference_points = new_reference_points.detach()

        return output


class SSNFuse(nn.Module):
    """ Fuse bert embeddings with query embeddings and predict boundaries """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim*2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, bert, query, mask):
        # bert: (bsz, len, 768)
        bert_embed = bert.unsqueeze(1).expand(-1, query.size(1), -1, -1)    # (bsz, q, len, 768)
        query_embed = query.unsqueeze(2).expand(-1, -1, bert.size(-2), -1)  # (bsz, q, len, 768)
        fuse = torch.cat([bert_embed, query_embed], dim=-1)                 # (bsz, q, len, 768*2)
        x = self.W(fuse)                                                    # (bsz, q, len, 768)

        x = self.v(torch.tanh(x)).squeeze(-1)                               # (bsz, q, len)
        mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
        x[~mask]=-1e25          # bsz
        x = x.softmax(dim=-1)                                               # (bsz, q, lens)

        return x


class SSNPromptAnchor(BertPreTrainedModel):
    def __init__(self, config: BertConfig, embed: torch.tensor, cls_token: int, entity_types: int,
                 prop_drop: float, freeze_transformer: bool, num_decoder_layers:int = 3, lstm_layers:int = 3, lstm_drop: float = 0.4, pos_size: int = 25, 
                 char_lstm_layers:int = 1, char_lstm_drop:int = 0.2, char_size:int = 25, use_glove: bool = True, use_pos:bool = True, use_char_lstm:bool = True, 
                 pool_type:str = "max", reduce_dim = False, bert_before_lstm = False, num_query: int=60):
        super(SSNPromptAnchor, self).__init__(config)
        # BERT model
        self.bert = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(config._name_or_path)    # genia
        self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_glove = use_glove
        self.use_pos = use_pos
        self.char_lstm_layers=char_lstm_layers
        self.char_lstm_drop=char_lstm_drop
        self.char_size=char_size
        self.use_char_lstm=use_char_lstm
        self.pool_type = pool_type
        self.reduce_dim = reduce_dim
        self.bert_before_lstm = bert_before_lstm
        
        lstm_input_size = 0
        if self.bert_before_lstm:
            lstm_input_size = config.hidden_size
        # use_glove or use_pos or use_char_lstm
        if use_glove:
            lstm_input_size += self.wordvec_size
        if use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(100, pos_size)
        if use_char_lstm:
            lstm_input_size += self.char_size * 2
            self.char_lstm = nn.LSTM(input_size = char_size, hidden_size = char_size, num_layers = char_lstm_layers,  bidirectional = True, dropout = char_lstm_drop, batch_first = True)
            self.char_embedding = nn.Embedding(103, char_size)


        if self.use_glove or self.use_pos or self.use_char_lstm or self.bert_before_lstm:
            lstm_hidden_size = lstm_input_size
            if self.bert_before_lstm:
                lstm_hidden_size = config.hidden_size//2
            self.lstm = nn.LSTM(input_size = lstm_input_size, hidden_size = lstm_hidden_size, num_layers = lstm_layers,  bidirectional = True, dropout = lstm_drop, batch_first = True)
            if self.reduce_dim or self.bert_before_lstm:
                if not self.bert_before_lstm:
                    self.reduce_dimension = nn.Linear(2 * lstm_input_size + config.hidden_size, config.hidden_size)
        query_dim = 4 
        
        # Decode
        self.query_embed = nn.Embedding(num_query, config.hidden_size)
        self.position_enc = PositionalEncoding(config.hidden_size, n_position=256)
        
        decoder_layer = SSNTransformerDecoderLayer(d_model=config.hidden_size, d_ffn=1024, dropout=0.1)
        self.decoder = SSNTransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers, query_dim=query_dim)

        self.entity_classifier = nn.Linear(config.hidden_size, entity_types)
        self.entity_left = SSNFuse(config.hidden_size)
        self.entity_right = SSNFuse(config.hidden_size)
        # add more span cls for discontinuous ner
        if config.discontinuous_ner:
            self.entity_left_second = SSNFuse(config.hidden_size)
            self.entity_right_second = SSNFuse(config.hidden_size)
            self.entity_left_third = SSNFuse(config.hidden_size)
            self.entity_right_third = SSNFuse(config.hidden_size)

        self.dropout = nn.Dropout(prop_drop)
        self._cls_token = cls_token
        self._entity_types = entity_types
        if not config.discontinuous_ner:
            self._init_query_emebdding()
        else:
            self._init_query_emebdding_discontinuous()
        
        # weight initialization
        self.init_weights()
        if use_glove:
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed)

        # refpoint
        self.refpoint_embed = nn.Embedding(self.query_embed.weight.shape[0], query_dim)
        # 均匀分布
        self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
        self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
        # 取消 x,y 的梯度，使得每张图片在输入到 Decoder 第一层时，使用的位置先验中心点(x,y)都是随机均匀分布的，
        # 而后每一层再由校正模块(bbox_embed)进行调整。
        # 这样可在一定程度上避免模型基于训练集而学到过份的归纳偏置(即过拟合)，更具泛化性
        self.refpoint_embed.weight.data[:, :2].requires_grad = False
        # MLP in anchor updating
        self.bbox_embed = MLP(config.hidden_size, config.hidden_size, query_dim, 3)
       
        
    def _init_query_emebdding(self):
        # prompt query objects
        query_input_ids, query_attention_mask, query_positions = self.get_prompt_query_objects()
        query_attention_mask = query_attention_mask.to(self.device)
        '''prompt -> encoder -> mask/mean -> decoder'''
        # get query objects
        with torch.no_grad():
            query_objects_states = self.bert(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
            )[0]             # [n_q, len, H]
            '''mask'''
            _, query_mask_ids = (query_input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            # [n_q, H]
            query_objects_states = query_objects_states[torch.arange(len(query_input_ids)), query_mask_ids]
        self.query_positions = query_positions
        self.query_embed.weight.data = query_objects_states
    
    def _init_query_emebdding_discontinuous(self):
            # prompt query objects
        query_input_ids, query_attention_mask, query_ent2span = self.get_prompt_query_objects_discontinuous()
        query_attention_mask = query_attention_mask.to(self.device)
        '''prompt -> encoder -> mask/mean -> decoder'''
        # get query objects
        with torch.no_grad():
            query_objects_states = self.bert(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
            )[0]             # [n_q, len, H]
            '''mask'''
            _, query_mask_ids = (query_input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            # [n_q, H]
            query_objects_states = query_objects_states[torch.arange(len(query_input_ids)), query_mask_ids]
        self.query_embed.weight.data = query_objects_states
        self.query_ent2span = query_ent2span
    
        
    def combine(self, sub, sup_mask, pool_type="max"):
        """ Combine different level representations """

        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup

    def _common_forward(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, token_masks_bool:torch.tensor, 
                       pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, 
                       char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):

        bsz = encodings.size(0)
        # encode
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h_token = self.bert(input_ids=encodings, attention_mask=context_masks)[0] # bsz, seq_len, hidden

        batch_size = encodings.shape[0]
        token_count = token_masks_bool.long().sum(-1,keepdim=True)
        h_token = self.combine(h_token, token_masks, self.pool_type)  # bsz, token_len, hidden
    
        embeds = []
        if self.bert_before_lstm:
            embeds = [h_token]
        if self.use_pos:
            pos_embed = self.pos_embedding(pos_encoding)
            pos_embed = self.dropout(pos_embed)
            embeds.append(pos_embed)
        if self.use_glove:
            word_embed = self.wordvec_embedding(wordvec_encoding)
            word_embed = self.dropout(word_embed)
            embeds.append(word_embed)
        if self.use_char_lstm:
            char_count = char_count.view(-1)
            token_masks_char = token_masks_char
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count*batch_size, max_char_count)
            char_encoding[char_count==0][:, 0] = 101
            char_count[char_count==0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input = char_embed, lengths = char_count.tolist(), enforce_sorted = False, batch_first = True)
            char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_size * 2)
            h_token_char = self.combine(char_embed, token_masks_char, self.pool_type)
            embeds.append(h_token_char)

        if len(embeds)>0:
            h_token_pos_wordvec_char = torch.cat(embeds, dim = -1)
            h_token_pos_wordvec_char_packed = nn.utils.rnn.pack_padded_sequence(input = h_token_pos_wordvec_char, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
            h_token_pos_wordvec_char_packed_o, (_, _) = self.lstm(h_token_pos_wordvec_char_packed)
            h_token_pos_wordvec_char, _ = nn.utils.rnn.pad_packed_sequence(h_token_pos_wordvec_char_packed_o, batch_first=True)

            rep = [h_token_pos_wordvec_char]
            if not self.bert_before_lstm:
                rep.append(h_token)
            h_token = torch.cat(rep, dim=-1)
            if self.reduce_dim and not self.bert_before_lstm:
                h_token = self.reduce_dimension(h_token)
                
        # context position
        pos = self.position_enc(h_token).repeat((bsz, 1, 1))
                
        # 准备 Decoder 的 query(包含内容&位置部分)
        # 位置 query：每张图片共享 n_q 个位置先验，其中每个都是 4d anchor box(x,y,w,h)
        # (n_q,4)->(bs, n_q, 4)
        refpoint_embed = self.refpoint_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # expand
        query_objects_states = self.query_embed.weight
        query_objects_states = query_objects_states.unsqueeze(0).expand((batch_size, -1, -1))  # [bs, n_q, H]
        
        # decode
        tgt = self.decoder(query_objects_states, h_token, pos, token_masks_bool, refpoint_embed)  # bsz, b_q, hidden
        
        # flat or nested ner:
        if not self.config.discontinuous_ner:
            entity_clf = self.entity_classifier(tgt)    # bsz, n_q, n_label
            entity_left = self.entity_left(h_token, tgt, token_masks_bool)      # bsz, n_q, token_len
            entity_right = self.entity_right(h_token, tgt, token_masks_bool)    # bsz, n_q, token_len
        # discontinuous ner
        else:
            # tgt: bsz, n_q, H, split to flat, two and three
            entity_clf, entity_left, entity_right = {}, {}, {}
            for key, start_end in self.query_ent2span.items():
                if key not in entity_clf:
                    entity_clf[key] = []
                    entity_left[key] = []
                    entity_right[key] = []
                start, end = start_end[0], start_end[1]
                # boundary predict
                # entity_left[key] = self.entity_left(h_token, tgt[:, start:end], token_masks_bool)    # bsz, n_key, token_len
                # entity_right[key] = self.entity_right(h_token, tgt[:, start:end], token_masks_bool)  # bsz, n_key, token_len
                # entity type predict
                if key == 'flat':
                    entity_left[key] = self.entity_left(h_token, tgt[:, start:end], token_masks_bool)    # bsz, n_key, token_len
                    entity_right[key] = self.entity_right(h_token, tgt[:, start:end], token_masks_bool)  # bsz, n_key, token_len
                    entity_clf[key] = self.entity_classifier(tgt[:, start:end])    # bsz, n_key, n_label
                elif key == 'two':
                    entity_left[key].append(self.entity_left(h_token, tgt[:, [start, start+2]], token_masks_bool))
                    entity_left[key].append(self.entity_left_second(h_token, tgt[:, [start+1, start+3]], token_masks_bool))
                    entity_right[key].append(self.entity_left_second(h_token, tgt[:, [start, start+2]], token_masks_bool))
                    entity_right[key].append(self.entity_right_second(h_token, tgt[:, [start+1, start+3]], token_masks_bool))
                    for i in range(start, end, 2):
                        entity_clf[key].append(self.entity_classifier(tgt[:, i:i+2].mean(dim=1)))    # bsz, n_two//2, n_label
                    entity_clf[key] = torch.stack(entity_clf[key], dim=1)
                    entity_left[key] = torch.cat(entity_left[key], dim=1)
                    entity_right[key] = torch.cat(entity_right[key], dim=1)
                elif key == 'three':
                    d_model = tgt.shape[-1]
                    entity_left[key].append(self.entity_left(h_token, tgt[:, start].view(bsz, -1, d_model), token_masks_bool))
                    entity_left[key].append(self.entity_left_second(h_token, tgt[:, start+1].view(bsz, -1, d_model), token_masks_bool))
                    entity_left[key].append(self.entity_left_third(h_token, tgt[:, start+2].view(bsz, -1, d_model), token_masks_bool))
                    entity_right[key].append(self.entity_right(h_token, tgt[:, start].view(bsz, -1, d_model), token_masks_bool))
                    entity_right[key].append(self.entity_right_second(h_token, tgt[:, start+1].view(bsz, -1, d_model), token_masks_bool))
                    entity_right[key].append(self.entity_right_third(h_token, tgt[:, start+2].view(bsz, -1, d_model), token_masks_bool))
                    for i in range(start, end, 3):
                        entity_clf[key].append(self.entity_classifier(tgt[:, i:i+2].mean(dim=1)))    # bsz, n_three//3, n_label
                    entity_clf[key] = torch.stack(entity_clf[key], dim=1)
                    entity_left[key] = torch.cat(entity_left[key], dim=1)
                    entity_right[key] = torch.cat(entity_right[key], dim=1)
        

        return entity_clf, (entity_left, entity_right)
    
        
    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, token_masks_bool:torch.tensor, 
                       pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, 
                       char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):
        """
        Do something different in training.

        """
        
        return self._common_forward(encodings, context_masks, token_masks, token_masks_bool, pos_encoding,
                        wordvec_encoding, char_encoding, token_masks_char, char_count)    
    
    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, token_masks_bool:torch.tensor, 
                       pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, 
                       char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):
        """
        Do something different in evaluation.

        """
        return self._common_forward(encodings, context_masks, token_masks, token_masks_bool, pos_encoding,
                        wordvec_encoding, char_encoding, token_masks_char, char_count)    
    
    
    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)
    
    def _get_template(self):
        return "The entity [MASK] is about {}."
    
    # # ace
    # def _read_entity2types(self):
    #     return {'FAC':'facility', 'WEA':'weapon', 'LOC':'location', 'VEH':'vehicle', 'GPE':'geographical', 'ORG':'organization', 'PER':'person'}

    # def _read_entity2num(self):
    #     return {'FAC':4, 'WEA':4, 'LOC':3, 'VEH':3, 'GPE':5, 'ORG':5, 'PER':8}
    
    
    def _read_entity2types(self):
        return {'ace05'
            'ADR':'adverse drug event'}
    
    def _read_entity2num(self):
        return {'ADR':13}
    
    def get_prompt_query_objects(self):
        # prepare query object inputs
        template = self._get_template()
        ent2type = self._read_entity2types()
        ent2num = self._read_entity2num()
        query_input_token = []
        query_positions = []
        for ent in ent2type:
            for i in range(ent2num[ent]):
                query_input_token.append(template.format(ent2type[ent]))
                query_positions.append(i)
        query_positions = torch.tensor(query_positions)
        query_inputs = self.tokenizer(query_input_token, padding=True, return_tensors='pt')
        query_input_ids, query_attention_mask = query_inputs['input_ids'], query_inputs['attention_mask']
        # get prompt entity 2num
        query_ent2span = {}
        ent_num_cums = torch.tensor(list(ent2num.values())).cumsum(dim=0)
        for i, ent in enumerate(ent2num.keys()):
            if i == 0:
                query_ent2span[ent] = (0, ent_num_cums[i].item()-1)
            else:
                query_ent2span[ent] = (ent_num_cums[i-1].item(), ent_num_cums[i].item()-1)
        return query_input_ids, query_attention_mask, query_positions

    def get_prompt_query_objects_discontinuous(self):
        # prepare query object inputs
        template = self._get_template()
        ent2type = self._read_entity2types()
        ent2num = self._read_entity2num()
        query_input_token = []
        for ent in ent2type:
            for i in range(ent2num[ent]):
                query_input_token.append(template.format(ent2type[ent]))
        query_inputs = self.tokenizer(query_input_token, padding=True, return_tensors='pt')
        query_input_ids, query_attention_mask = query_inputs['input_ids'], query_inputs['attention_mask']
        # get prompt entity 2num
        query_ent2span = {'flat': (0, 6), 'two': (6, 10), 'three': (10, 13)}
        return query_input_ids, query_attention_mask, query_ent2span

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor, d_model):
    # bsz, n_q, 4 = pos_tensor.size()
    # sineembed_tensor = torch.zeros(bsz, n_q, d_model)
    scale = 2 * math.pi
    dim_t = torch.arange(d_model, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / d_model)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)