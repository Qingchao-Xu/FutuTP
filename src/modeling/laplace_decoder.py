from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List, Tuple, NamedTuple, Any
import numpy as np
from utils_files import utils, config
from utils_files.utils import init_weights, get_dis, get_dis_list
from modeling.vectornet import *
from modeling.motion_refinement import trajectory_refinement
from utils_files.loss import *


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class GRUDecoder(nn.Module):
    def __init__(self, args, vectornet) -> None:
        super(GRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.input_size = args.hidden_size
        self.hidden_size = args.hidden_size
        self.future_steps = args.future_frame_num
        self.num_modes = args.mode_num
        self.min_scale = min_scale
        self.args = args
        self.dense = args.future_frame_num
        self.z_size = args.z_size
        self.smothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 1))
        self.aggregate_global_z = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.reg_loss = LaplaceNLLLoss(reduction='none')
        self.vel_encoding_layer = nn.Sequential(nn.Linear(32, self.hidden_size, bias=True),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size, self.hidden_size, bias=True))
        self.proj_vel = nn.Sequential(nn.Linear(19 * self.hidden_size, self.num_modes * self.hidden_size),
                                      nn.LayerNorm(self.num_modes * self.hidden_size),
                                      nn.ReLU(inplace=True))
        self.proj_local = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                        nn.LayerNorm(self.hidden_size),
                                        nn.ReLU(inplace=True))
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='none')
        if "step_lane_score" in args.other_params:
            self.multihead_proj_global = nn.Sequential(
                                        nn.Linear(self.hidden_size*2, self.num_modes * self.hidden_size),
                                        nn.LayerNorm(self.num_modes * self.hidden_size),
                                        nn.ReLU(inplace=True))  
            decoder_layer_dense_label = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.dense_label_cross_attention = nn.TransformerDecoder(decoder_layer_dense_label, num_layers=1)
            self.dense_lane_decoder = DecoderResCat(self.hidden_size, self.hidden_size * 3, out_features=self.dense)
            self.proj_topk = MLP(self.hidden_size+1, self.hidden_size)
            decoder_layer_aggregation = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.aggregation_cross_att = nn.TransformerDecoder(decoder_layer_aggregation, num_layers=1)
        else:
            self.multihead_proj_global = nn.Sequential(
                                        nn.Linear(self.hidden_size, self.num_modes * self.hidden_size),
                                        nn.LayerNorm(self.num_modes * self.hidden_size),
                                        nn.ReLU(inplace=True))
        self.apply(init_weights)   
        if "stage_two" in args.other_params:
            if args.do_train:
                model_recover = torch.load(args.other_params['stage-two-train_recover'])
                vectornet.decoder = self
                utils.load_model(vectornet, model_recover)
            self.trajectory_refinement = trajectory_refinement(args)

    def dense_lane_aware(self, i, mapping, lane_states_batch, lane_states_length, element_hidden_states, \
                            element_hidden_states_lengths, global_hidden_states, device, loss):
        def dense_lane_scores():
            lane_states_batch_attention = lane_states_batch + self.dense_label_cross_attention(
                lane_states_batch, element_hidden_states.unsqueeze(0), tgt_key_padding_mask=src_attention_mask_lane)
            dense_lane_scores = self.dense_lane_decoder(torch.cat([global_hidden_states.unsqueeze(0).expand(
                lane_states_batch.shape), lane_states_batch, lane_states_batch_attention], dim=-1))
            dense_lane_scores = F.log_softmax(dense_lane_scores, dim=0)
            return dense_lane_scores
        max_vector_num = lane_states_batch.shape[1]
        batch_size = len(mapping)
        src_attention_mask_lane = torch.zeros([batch_size, lane_states_batch.shape[1]], device=device)
        for i in range(batch_size):
            assert lane_states_length[i] > 0
            src_attention_mask_lane[i, :lane_states_length[i]] = 1
        src_attention_mask_lane = src_attention_mask_lane == 0
        lane_states_batch = lane_states_batch.permute(1, 0, 2)
        dense_lane_pred = dense_lane_scores()
        dense_lane_pred = dense_lane_pred.permute(1, 0, 2)
        lane_states_batch = lane_states_batch.permute(1, 0, 2)
        dense = self.dense
        dense_lane_pred = dense_lane_pred.permute(0, 2, 1)
        dense_lane_pred = dense_lane_pred.contiguous().view(-1, max_vector_num)
        if self.args.do_train:
            dense_lane_targets = torch.zeros([batch_size, dense], device=device, dtype=torch.long)
            for i in range(batch_size):
                dense_lane_targets[i, :] = torch.tensor(np.array(mapping[i]['dense_lane_labels']), dtype=torch.long, device=device)
            loss_weight = self.args.lane_loss_weight
            dense_lane_targets = dense_lane_targets.view(-1)
            loss += loss_weight*F.nll_loss(dense_lane_pred, dense_lane_targets, reduction='none').\
                    view(batch_size, dense).sum(dim=1)
        mink = self.args.topk
        dense_lane_topk = torch.zeros((dense_lane_pred.shape[0], mink, self.hidden_size), device=device)
        dense_lane_topk_scores = torch.zeros((dense_lane_pred.shape[0], mink), device=device)
        neighbor_lane_batch = []
        neighbor_lane_score_batch = []
        for i in range(dense_lane_topk_scores.shape[0]):
            idxs_lane = i // dense
            k = min(mink, lane_states_length[idxs_lane])
            _, idxs_topk = torch.topk(dense_lane_pred[i], k)
            if (i + 1) % self.args.future_frame_num == 0:
                lane_ids = []
                for idx in range(k):
                    if idxs_topk[idx] >= len(mapping[idxs_lane]['polygons']):
                        continue
                    lane_pos = 0
                    for polygon in enumerate(mapping[idxs_lane]['polygons'][idxs_topk[idx]]):
                        lane_pos += polygon[1]
                    if len(mapping[idxs_lane]['polygons'][idxs_topk[idx]]) > utils.eps:
                        lane_pos = lane_pos / len(mapping[idxs_lane]['polygons'][idxs_topk[idx]])
                    else:
                        continue
                    for polygon in enumerate(mapping[idxs_lane]['polygons']):
                        polygon_pos = np.array(polygon[1])
                        if "nuscenes" in self.args.other_params:
                            temp_dist = np.argmin([min(get_dis_list(polygon[1], lane_pos))])
                        else:
                            temp_dist = np.min(get_dis(polygon_pos, lane_pos))
                        if temp_dist < 5:
                            lane_ids.append(polygon[0])
                neighbor_lane_batch.append(lane_states_batch[idxs_lane, lane_ids])
                neighbor_lane_score_batch.append(dense_lane_pred[i][lane_ids].unsqueeze(1))
            dense_lane_topk[i][:k] = lane_states_batch[idxs_lane, idxs_topk]
            dense_lane_topk_scores[i][:k] = dense_lane_pred[i][idxs_topk]
        neighbor_lane_batch, neighbor_lane_length = utils.merge_tensors(neighbor_lane_batch, device=device)
        neighbor_lane_score_batch, neighbor_lane_score_length = utils.merge_tensors(neighbor_lane_score_batch, device=device, hidden_size=1)
        neighbor_lane = torch.cat([neighbor_lane_batch, neighbor_lane_score_batch], dim=-1)
        dense_lane_topk = torch.cat([dense_lane_topk, dense_lane_topk_scores.unsqueeze(-1)], dim=-1)
        dense_lane_topk = dense_lane_topk.view(batch_size, dense*mink, self.hidden_size + 1)
        dense_lane_topk = torch.cat([dense_lane_topk, neighbor_lane], dim=1)
        neighbor_lane_length = [i + dense*mink for i in neighbor_lane_length]
        return dense_lane_topk, neighbor_lane_length

    def forward(self, pred_dense_trajs_batch, mapping: List[Dict], batch_size, lane_states_batch, lane_states_length, inputs: Tensor,
                inputs_lengths: List[int], hidden_states: Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_steps])
        local_embed = inputs[:, 0, :]
        global_embed = hidden_states[:, 0, :]

        if "step_lane_score" in self.args.other_params:
            dense_lane_topk, neighbor_lane_length = self.dense_lane_aware\
            (0, mapping, lane_states_batch, lane_states_length, local_embed, inputs_lengths, global_embed, device, loss)
            src_attention_mask_lane = torch.zeros([batch_size, dense_lane_topk.shape[1]], device=device)
            for i in range(batch_size):
                assert neighbor_lane_length[i] > 0
                src_attention_mask_lane[i, :neighbor_lane_length[i]] = 1
            src_attention_mask_lane = src_attention_mask_lane == 0
            dense_lane_topk = dense_lane_topk.permute(1, 0, 2)
            dense_lane_topk = self.proj_topk(dense_lane_topk)
            global_embed_att = global_embed + self.aggregation_cross_att(global_embed.unsqueeze(0),
                                                                         dense_lane_topk,
                                                                         memory_key_padding_mask=src_attention_mask_lane).squeeze(0)
            global_embed = torch.cat([global_embed, global_embed_att], dim=-1)

        local_embed = local_embed.repeat(self.num_modes, 1, 1)
        global_embed = self.multihead_proj_global(global_embed).view(-1, self.num_modes, self.hidden_size)
        batch_size = global_embed.shape[0]
        global_embed = global_embed.transpose(0, 1)

        pi = self.pi(torch.cat((local_embed, global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)

        z_size = self.z_size
        z = torch.randn(self.num_modes*batch_size,  z_size, device=device)
        global_embed = torch.cat([global_embed, z], dim=-1)
        global_embed = self.aggregate_global_z(global_embed)
        
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)
        local_embed = local_embed.reshape(-1, self.input_size).unsqueeze(0)
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)
        scale = F.elu_(self.scale(out), alpha=1.0)+ 1.0 + self.min_scale
        scale = scale.view(self.num_modes, -1, self.future_steps, 2)
        if "stage_two" in self.args.other_params:
            past_traj = utils.get_from_mapping(mapping, 'past_traj')
            past_traj = torch.tensor(np.array(past_traj), dtype=torch.float32, device=device)
            past_traj = past_traj[:,:,:2]
            past_traj = past_traj.expand(self.num_modes, *past_traj.shape)
            full_traj = torch.cat((past_traj, loc), dim=2)
            loc_delta, _ = self.trajectory_refinement(out, full_traj, global_embed, local_embed)
        if "stage_two" in self.args.other_params:
            return self.laplace_decoder_loss((loc, loc_delta, past_traj), scale, pi, labels_is_valid, loss, DE, device, labels, mapping, pred_dense_trajs_batch)
        else:
            return self.laplace_decoder_loss(loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping, pred_dense_trajs_batch)


    def laplace_decoder_loss(self, loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping=None, pred_dense_trajs_batch=None):
        if "stage_two" in self.args.other_params:
            original_loc, loc_delta, past_traj = loc
            loc = original_loc + loc_delta
        y_hat = torch.cat((loc, scale), dim=-1)
        batch_size = y_hat.shape[1]
        labels = torch.tensor(np.array(labels), device = device)
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - labels, p=2, dim=-1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
        future_loss = 0
        if "stage_two" in self.args.other_params and self.args.do_train:
            loc_delta_best = loc_delta[best_mode, torch.arange(y_hat.shape[1])]
            delta_label = labels-original_loc[best_mode, torch.arange(y_hat.shape[1])]
            reg_delta_loss = torch.norm(loc_delta_best-delta_label, p=2, dim=-1)
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1) + 5*reg_delta_loss
            loss += get_angle_diff(labels, y_hat_best[:, :, :2], past_traj)*2
            soft_target = F.softmax(-l2_norm/ self.future_steps, dim=0).t().detach()
            cls_loss = self.cls_loss(pi, soft_target)
        else:
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1)
            soft_target = F.softmax(-l2_norm/ self.future_steps, dim=0).t().detach()
            cls_loss = self.cls_loss(pi, soft_target)

        if self.args.do_train:
            for i in range(batch_size):
                future_label = torch.tensor(mapping[i]['dense_agent_labels'], device=device).to(torch.float32)
                dense_future_loss = 10 * F.smooth_l1_loss(pred_dense_trajs_batch[i], future_label, reduction='none').mean()
                loss[i] += dense_future_loss
                future_loss += dense_future_loss.item()

        if self.args.do_train:
            for i in range(batch_size):
                if self.args.do_train:
                    assert labels_is_valid[i][-1]
                loss_ = reg_loss[i]
                loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_steps, 1)
                if labels_is_valid[i].sum() > utils.eps:
                    loss[i] += loss_.sum() / labels_is_valid[i].sum()
                loss[i] += cls_loss[i]

        if self.args.do_eval:
            outputs = loc.permute(1, 0, 2, 3).detach()
            pred_probs = F.softmax(pi, dim=-1).cpu().detach().numpy()
            for i in range(batch_size):
                if self.args.visualize:
                    labels = utils.get_from_mapping(mapping, 'labels')
                    labels = np.array(labels)
                    predict_endpoint = outputs[i].cpu().numpy()[:, -1, :]
                    label_endpoint = labels[i][-1, :]
                    errors = np.sqrt(np.square(predict_endpoint[:, 0] - label_endpoint[0]) + np.square(
                        predict_endpoint[:, 1] - label_endpoint[1]))
                    fde = np.min(errors)
                    index = np.argmin(errors)
                    utils.visualize_gifs(index, mapping[i], self.args.future_frame_num, labels=labels[i], predict=outputs[i].cpu().numpy())
                outputs[i] = utils.to_origin_coordinate(outputs[i], i)
                if "vis_nuscenes" in self.args.other_params:
                    from utils_files import vis_nuscenes
                    vis_nuscenes.generate_nuscenes_gif(mapping[i], self.args.future_frame_num, outputs[i].cpu().numpy())
            outputs = outputs.cpu().numpy()
            return outputs, pred_probs, None
        return loss.mean(), DE, None, future_loss
