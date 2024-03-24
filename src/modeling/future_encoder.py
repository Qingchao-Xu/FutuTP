from typing import Dict, List
import torch
from torch import nn
from utils_files import utils
from .global_graph import GlobalGraphRes

class FutureEncoder(nn.Module):
    def __init__(self, args, hidden_size):
        super(FutureEncoder, self).__init__()
        self.hidden_size = hidden_size
        if "nuscenes" in args.other_params:
            num_layers = 1
        else:
            num_layers = 3
        decoder_layer_A2A = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size)
        self.laneGCN_A2A = nn.TransformerDecoder(decoder_layer_A2A, num_layers=num_layers)

    def forward(self, lane_states_batch, lane_states_length, global_hidden_states, mapping, device):
        agent_num = global_hidden_states.size(1) - lane_states_batch.size(1)
        agent_states_batch = global_hidden_states[:, 0:agent_num, :]
        lane_states_batch2 = global_hidden_states[:, agent_num:, :]

        agent_states_batch = agent_states_batch + self.laneGCN_A2A(agent_states_batch, agent_states_batch)
        global_hidden_states = torch.cat([agent_states_batch, lane_states_batch2], dim=1)

        return global_hidden_states
