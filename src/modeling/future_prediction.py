from typing import Dict, List
import torch
from torch import nn
from utils_files import utils


class FuturePrediction(nn.Module):
    def __init__(self, hidden_size, num_future_frames):
        super(FuturePrediction, self).__init__()
        self.hidden_size = hidden_size
        self.num_future_frames = num_future_frames
        self.obj_pos_encoding_layer = nn.Sequential(nn.Linear(2, hidden_size, bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_size, hidden_size, bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_size, hidden_size, bias=True))
        self.dense_future_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=False),
                                               nn.LayerNorm(hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size, bias=False),
                                               nn.LayerNorm(hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size, num_future_frames * 2, bias=True))
        self.future_traj_mlps = nn.Sequential(nn.Linear(2 * num_future_frames, hidden_size, bias=True),
                                              nn.ReLU(),
                                              nn.Linear(hidden_size, hidden_size, bias=True),
                                              nn.ReLU(),
                                              nn.Linear(hidden_size, hidden_size, bias=True))
        self.traj_fusion_mlps = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size, bias=True),
                                              nn.ReLU(),
                                              nn.Linear(hidden_size, hidden_size, bias=True),
                                              nn.ReLU(),
                                              nn.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, global_hidden_states, mapping, device):
        batch_size = len(mapping)
        pred_dense_trajs_batch = []
        for i in range(batch_size):
            obj_trajs = torch.tensor(mapping[i]['dense_agent_trajs'],
                                     device=device).to(torch.float32)
            obj_pos = obj_trajs[:, -1, :]
            obj_pos_feature = self.obj_pos_encoding_layer(obj_pos)
            obj_feature = global_hidden_states[i][mapping[i]['dense_agent_ids']]
            obj_fused_feature = torch.cat((obj_pos_feature, obj_feature), dim=-1)
            pred_dense_trajs = self.dense_future_head(obj_fused_feature)
            pred_dense_trajs = pred_dense_trajs.view(pred_dense_trajs.shape[0], self.num_future_frames, 2)
            pred_dense_trajs = pred_dense_trajs[:, :, 0:2] + obj_pos[:, None, 0:2]
            pred_dense_trajs_batch.append(pred_dense_trajs)
            obj_future_input = pred_dense_trajs.flatten(start_dim=1, end_dim=2)
            obj_future_feature = self.future_traj_mlps(obj_future_input)
            obj_full_trajs_feature = torch.cat((obj_feature, obj_future_feature), dim=-1)

            obj_feature2 = obj_feature + self.traj_fusion_mlps(obj_full_trajs_feature)
            global_hidden_states[i][mapping[i]['dense_agent_ids']] = obj_feature2

        return global_hidden_states, pred_dense_trajs_batch
