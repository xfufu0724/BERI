
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from icecream import ic



class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, )

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, sub_embed, obj_embed, pos_embed, entity_embed,rel_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        #decoder
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt= torch.zeros_like(query_embed)
        
        #decoder
        sub_embed = sub_embed.unsqueeze(1).repeat(1, bs, 1)
        obj_embed = obj_embed.unsqueeze(1).repeat(1, bs, 1)
        sub_tgt = torch.zeros_like(sub_embed)
        obj_tgt = torch.zeros_like(obj_embed)

        entity_embed =entity_embed.unsqueeze(1).repeat(1, bs, 1)
        rel_embed =rel_embed.unsqueeze(1).repeat(1, bs, 1)
        rel_tgt = torch.zeros_like(rel_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs,hs_rel, hs_sub, hs_obj= self.decoder(tgt, memory, sub_tgt,obj_tgt, rel_tgt,
                                                 memory_key_padding_mask=mask,pos=pos_embed, 
                                                 entity_pos = entity_embed,sub_pos = sub_embed, obj_pos = obj_embed,
                                                 rel_pos=rel_embed)


        return  hs.transpose(1, 2),hs_rel.transpose(1, 2), hs_sub.transpose(1, 2), hs_obj.transpose(1, 2)



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_dyt=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.use_dyt = use_dyt
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 =create_norm_layer(d_model)
        self.norm2 =create_norm_layer(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self,tgt, memory,sub_tgt,obj_tgt,rel_tgt,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, entity_pos: Optional[Tensor] = None,
                sub_pos: Optional[Tensor] = None, obj_pos: Optional[Tensor] = None, rel_pos: Optional[Tensor] = None):

        output_tgt = tgt
        output_sub = sub_tgt
        output_obj = obj_tgt
        output_rel = rel_tgt
        intermediate_entity = []
        intermediate_sub = []
        intermediate_obj = []
        intermediate_rel = []

        final_sub_loc_weights = None
        final_obj_loc_weights = None
        final_sub_obj_weights = None
        final_obj_sub_weights = None
        final_rel_cross_weights = None


        for layer in self.layers:

            output_entity,output_rel, output_sub, output_obj, sub_loc_weights, obj_loc_weights, 
            sub_obj_weights, obj_sub_weights, rel_cross_weights = layer(
                output_tgt, memory, output_sub, output_obj, output_rel,
                entity_pos, sub_pos, obj_pos, rel_pos,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask, pos=pos
)


            if self.return_intermediate:
                intermediate_entity.append(output_entity)
                intermediate_rel.append(output_rel)
                intermediate_sub.append(output_sub)
                intermediate_obj.append(output_obj)

        
        
        
        if self.return_intermediate:
            return (
                torch.stack(intermediate_entity), 
                torch.stack(intermediate_rel), 
                torch.stack(intermediate_sub), 
                torch.stack(intermediate_obj),\
               final_sub_loc_weights, \
               final_obj_loc_weights, \
               final_sub_obj_weights, \
               final_obj_sub_weights, \
               final_rel_cross_weights
            )
        
        # Return the final outputs along with attention weights
        return  output_entity, output_rel, output_sub, output_obj,\
        

class TransformerDecoderLayer(nn.Module):
    """triplet decoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", use_dyt=False):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        
        self.nhead = nhead
        self.dropout = dropout
        self.d_model = d_model
        self.use_dyt = use_dyt


        self.self_attn_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2_entity = nn.Dropout(dropout)
        self.norm2_entity = create_norm_layer(d_model)

        self.cross_attn_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1_entity = nn.Dropout(dropout)
        self.norm1_entity =create_norm_layer(d_model)
        

        self.shared_entity_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_entity = nn.Dropout(dropout)
        self.norm_entity = create_norm_layer(d_model)

        self.dropout2_rel = nn.Dropout(dropout)
        self.norm2_rel =create_norm_layer(d_model)

        self.linear1_entity = nn.Linear(d_model, dim_feedforward)
        self.dropout3_entity = nn.Dropout(dropout)
        self.linear2_entity = nn.Linear(dim_feedforward, d_model)
        self.dropout4_entity = nn.Dropout(dropout)
        self.norm3_entity = create_norm_layer(d_model)

        self.linear1_sub = nn.Linear(d_model, dim_feedforward)
        self.dropout3_sub = nn.Dropout(dropout)
        self.linear2_sub = nn.Linear(dim_feedforward, d_model)
        self.dropout4_sub = nn.Dropout(dropout)
        self.norm3_sub = create_norm_layer(d_model)

        self.linear1_obj = nn.Linear(d_model, dim_feedforward)
        self.dropout3_obj = nn.Dropout(dropout)
        self.linear2_obj = nn.Linear(dim_feedforward, d_model)
        self.dropout4_obj = nn.Dropout(dropout)
        self.norm3_obj = create_norm_layer(d_model)


        self.linear1_rel = nn.Linear(d_model, dim_feedforward)
        self.dropout3_rel = nn.Dropout(dropout)
        self.linear2_rel = nn.Linear(dim_feedforward, d_model)
        self.dropout4_rel = nn.Dropout(dropout)
        self.norm3_rel = create_norm_layer(d_model)

    
    def forward_ffn_entity(self, tgt):
        tgt2 = self.linear2_entity(self.dropout3_entity(self.activation(self.linear1_entity(tgt))))
        tgt = tgt + self.dropout4_entity(tgt2)
        tgt = self.norm3_entity(tgt)
        return tgt
    
    def forward_ffn_sub(self, tgt_sub):
        tgt_sub_1 = self.linear2_sub(self.dropout3_sub(self.activation(self.linear1_sub(tgt_sub))))
        tgt_sub = tgt_sub + self.dropout4_sub(tgt_sub_1)
        tgt_sub = self.norm3_sub(tgt_sub)
        return tgt_sub

    def forward_ffn_obj(self, tgt_obj):
        tgt_obj_1 = self.linear2_obj(self.dropout3_obj(self.activation(self.linear1_obj(tgt_obj))))
        tgt_obj = tgt_obj + self.dropout4_obj(tgt_obj_1)
        tgt_obj = self.norm3_obj(tgt_obj)
        return tgt_obj

    def forward_ffn_rel(self,tgt_rel):
        tgt_rel_1 = self.linear2_rel(self.dropout3_rel(self.activation(self.linear1_rel(tgt_rel))))
        tgt_rel = tgt_rel + self.dropout4_rel(tgt_rel_1)
        tgt_rel = self.norm3_rel(tgt_rel)
        return tgt_rel
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def co_attention(self, sub_tgt, obj_tgt, nhead, dropout=0.1):
        
        device = sub_tgt.device
        sub_to_obj_attention = nn.MultiheadAttention(self.d_model, nhead, dropout=dropout).to(device)
        obj_to_sub_attention = nn.MultiheadAttention(self.d_model, nhead, dropout=dropout).to(device)

        # 主体关注客体
        sub_to_obj_attn, sub_to_obj_weight = sub_to_obj_attention(
            query=sub_tgt,
            key=obj_tgt,
            value=obj_tgt
        )
        sub_to_obj_attn = sub_tgt + F.dropout(sub_to_obj_attn, p=dropout)
        o_obj_attn = F.layer_norm(sub_to_obj_attn, sub_to_obj_attn.shape[-1:])

        # 客体关注主体
        obj_to_sub_attn, obj_to_sub_weight = obj_to_sub_attention(
            query=obj_tgt,
            key=sub_tgt,
            value=sub_tgt
        )
        obj_to_sub_attn = obj_tgt + F.dropout(obj_to_sub_attn, p=dropout)
        
        obj_to_sub_attn = F.layer_norm(obj_to_sub_attn, obj_to_sub_attn.shape[-1:])


        return sub_to_obj_attn, obj_to_sub_attn
    
    def forward(self, tgt, memory,sub_tgt,obj_tgt, rel_embed,entity_pos,sub_pos,obj_pos,
                rel_pos,tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,):

        q = k = self.with_pos_embed(tgt, entity_pos)
        tgt2 = self.self_attn_entity(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout2_entity(tgt2)
        tgt = self.norm2_entity(tgt)
        
        q_encoder = k_encoder = self.with_pos_embed(memory, pos)
        tgt2 = self.cross_attn_entity(query=self.with_pos_embed(tgt, entity_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout1_entity(tgt2)
        tgt = self.norm1_entity(tgt)
        tgt_entity = self.forward_ffn_entity(tgt)

        q_sub_att  = self.with_pos_embed(sub_tgt, sub_pos)
        q_obj_att  = self.with_pos_embed(obj_tgt, obj_pos)
        
        shared_entity_sub, sub_loc_weights = self.shared_entity_attn(
            query=q_sub_att,
            key=k_encoder,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt_sub = sub_tgt + self.dropout_entity(shared_entity_sub)
        tgt_sub = self.norm_entity(tgt_sub)
        tgt_sub = self.forward_ffn_sub(tgt_sub)
        
        
        shared_entity_obj, obj_loc_weights = self.shared_entity_attn(
            query=q_obj_att,
            key=k_encoder,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt_obj = obj_tgt + self.dropout_entity(shared_entity_obj)
        tgt_obj = self.norm_entity(tgt_obj)
        tgt_obj = self.forward_ffn_obj(tgt_obj)

        tgt_sub_pos = self.with_pos_embed(tgt_sub,sub_pos)
        tgt_obj_pos = self.with_pos_embed(tgt_obj,obj_pos)
        sub_co, obj_co, sub_att, obj_att = self.co_attention(tgt_sub_pos, tgt_obj_pos, self.nhead, self.dropout)
        entity = torch.cat([sub_co,obj_co], dim=0)
    
        
        rel_attn,rel_cross_weights = self.shared_entity_attn(query = self.with_pos_embed(rel_embed, rel_pos),
                                 key = entity, value = entity)
        rel_attn = rel_embed + self.dropout2_rel(rel_attn)
        rel_attn = self.norm2_rel(rel_attn)
        rel_feat = self.forward_ffn_rel(rel_attn)

        return tgt_entity, rel_feat, tgt_sub, tgt_obj,sub_loc_weights,obj_loc_weights,sub_att,obj_att,rel_cross_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

