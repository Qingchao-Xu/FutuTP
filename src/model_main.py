from typing import Dict, List
import torch
from torch import nn
from modeling.vectornet import VectorNet
from modeling.global_graph import CrossAttention, GlobalGraphRes
from modeling.laplace_decoder import  GRUDecoder
from modeling.future_prediction import FuturePrediction
from modeling.future_encoder import FutureEncoder
from utils_files import utils, config
import torch.nn.functional as F
import numpy as np


class ModelMain(nn.Module):

    def __init__(self, args_: config.Args):
        super(ModelMain, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.encoder = VectorNet(args)
        self.global_graph = GlobalGraphRes(hidden_size)
        self.dense_future_prediction = FuturePrediction(hidden_size, args.future_frame_num)
        self.future_encoder = FutureEncoder(args, hidden_size)
        self.decoder = GRUDecoder(args, self)

    def forward(self, mapping: List[Dict], device):
        vector_matrix = utils.get_from_mapping(mapping, 'matrix')
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')
        batch_size = len(vector_matrix)
        utils.batch_origin_init(mapping)

        # Encoder
        all_element_states_batch, lane_states_batch1 = self.encoder.forward(mapping, vector_matrix, polyline_spans, device, batch_size)
        # Global interacting operation
        inputs, inputs_lengths = utils.merge_tensors(all_element_states_batch, device=device)
        lane_states_batch1, lane_states_length = utils.merge_tensors(lane_states_batch1, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)
        global_hidden_states1 = self.global_graph(inputs, attention_mask, mapping)
        # Future Interaction
        global_hidden_states2, pred_dense_trajs_batch = self.dense_future_prediction(global_hidden_states1, mapping, device=device)
        global_hidden_states3 = self.future_encoder(lane_states_batch1, lane_states_length, global_hidden_states2, mapping, device)
        global_hidden_states3 = global_hidden_states1 + global_hidden_states3
        # Decoder
        return self.decoder(pred_dense_trajs_batch, mapping, batch_size, lane_states_batch1, lane_states_length, inputs, inputs_lengths, global_hidden_states3, device)


    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict_rename_key = {}
        for key in state_dict.keys():
            if key.startswith('point_level_') or key.startswith('laneGCN_'):
                state_dict_rename_key['encoder.'+key] = state_dict[key]
            else:
                state_dict_rename_key[key] = state_dict[key]
        super(ModelMain, self).load_state_dict(state_dict_rename_key, strict)
