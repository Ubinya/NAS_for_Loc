import torch

from utils.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import  models.cell_level_search
from models.cell_level_search import PRIMITIVES
import cell_level_search
from models.operations import *
from models.decoding_formulas import Decoder
import pdb
from operations import *


def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0]+1)/2, sim_nn.new_tensor(0))
    return matches, scores


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighbor(BaseModel):
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        pass

    def _forward(self, data):
        sim = torch.einsum(
            'bdn,bdm->bnm', data['descriptors0'], data['descriptors1'])
        matches0, scores0 = find_nn(
            sim, self.conf['ratio_threshold'], self.conf['distance_threshold'])
        if self.conf['do_mutual_check']:
            matches1, scores1 = find_nn(
                sim.transpose(1, 2), self.conf['ratio_threshold'],
                self.conf['distance_threshold'])
            matches0 = mutual_check(matches0, matches1)
        return {
            'matches0': matches0,
            'matching_scores0': scores0,
        }


class NASMatcher(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def _init(self, conf):
        self.net = MatcherSearch(conf)

    def _forward(self, data):
        return self.net(data)


class MatcherSearch(BaseModel):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }# from SuperGlue

    def __init__(self, num_layers, filter_multiplier=8, block_multiplier=4, step=4, cell=cell_level_search.Cell):
        super(MatcherSearch, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._initialize_alphas_betas()
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = f_initial * self._block_multiplier

        print('Feature Net block_multiplier:{0}'.format(block_multiplier))
        print('Feature Net filter_multiplier:{0}'.format(filter_multiplier))
        print('Feature Net f_initial:{0}'.format(f_initial))

        """ConvBR(C_in, C_out, kernel_size, stride, padding, bn=True, relu=True)"""
        self.stem0 = ConvBR(3, half_f_initial * self._block_multiplier, 3, stride=1, padding=1)
        self.stem1 = ConvBR(half_f_initial * self._block_multiplier, half_f_initial * self._block_multiplier, 3, stride=3, padding=1)
        self.stem2 = ConvBR(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, 3, stride=1, padding=1)

        for i in range(self._num_layers):
            if i == 0:
                """Cell(steps, block_multiplier, prev_prev_fmultiplier,
                 prev_fmultiplier_down, prev_fmultiplier_same, prev_fmultiplier_up,
                 filter_multiplier)"""
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.last_3 = ConvBR(self._num_end, self._num_end, 1, 1, 0, bn=False, relu=False)
        self.last_6 = ConvBR(self._num_end * 2, self._num_end, 1, 1, 0)
        self.last_12 = ConvBR(self._num_end * 4, self._num_end * 2, 1, 1, 0)
        self.last_24 = ConvBR(self._num_end * 8, self._num_end * 4, 1, 1, 0)



    def forward(self, x):
        #------------follow from build_model_2d---------------
        self.level_3 = []
        self.level_6 = []
        self.level_12 = []
        self.level_24 = []

        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)

        self.level_3.append(stem2)

        count = 0
        normalized_betas = torch.randn(self._num_layers, 4, 3).cuda()
        # Softmax on alphas and betas
        if torch.cuda.device_count() > 1:
            # print('more than 1 gpu used!')
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(self.alphas.to(device=img_device), dim=-1)

            # normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘]
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device),
                                                               dim=-1) * (2 / 3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device),
                                                               dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device),
                                                               dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device),
                                                               dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:1].to(device=img_device),
                                                               dim=-1) * (2 / 3)

        else:
            normalized_alphas = F.softmax(self.alphas, dim=-1)

            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1) * (2 / 3)

        for layer in range(self._num_layers):

            if layer == 0:
                level3_new, = self.cells[count](None, None, self.level_3[-1], None, normalized_alphas)
                count += 1
                level6_new, = self.cells[count](None, self.level_3[-1], None, None, normalized_alphas)
                count += 1

                level3_new = normalized_betas[layer][0][1] * level3_new
                level6_new = normalized_betas[layer][0][2] * level6_new
                self.level_3.append(level3_new)
                self.level_6.append(level6_new)

            elif layer == 1:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2 = self.cells[count](None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               None,
                                                               normalized_alphas)
                count += 1
                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][2] * level6_new_2

                level12_new, = self.cells[count](None,
                                                 self.level_6[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level12_new = normalized_betas[layer][1][2] * level12_new
                count += 1

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)

            elif layer == 2:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2, level6_new_3 = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             self.level_12[-1],
                                                                             normalized_alphas)
                count += 1
                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][
                    1] * level6_new_2 + normalized_betas[layer][2][
                                 0] * level6_new_3

                level12_new_1, level12_new_2 = self.cells[count](None,
                                                                 self.level_6[-1],
                                                                 self.level_12[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level12_new = normalized_betas[layer][1][2] * level12_new_1 + normalized_betas[layer][2][
                    1] * level12_new_2

                level24_new, = self.cells[count](None,
                                                 self.level_12[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level24_new = normalized_betas[layer][2][2] * level24_new
                count += 1

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            elif layer == 3:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2, level6_new_3 = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             self.level_12[-1],
                                                                             normalized_alphas)
                count += 1
                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][
                    1] * level6_new_2 + normalized_betas[layer][2][
                                 0] * level6_new_3

                level12_new_1, level12_new_2, level12_new_3 = self.cells[count](self.level_12[-2],
                                                                                self.level_6[-1],
                                                                                self.level_12[-1],
                                                                                self.level_24[-1],
                                                                                normalized_alphas)
                count += 1
                level12_new = normalized_betas[layer][1][2] * level12_new_1 + normalized_betas[layer][2][
                    1] * level12_new_2 + normalized_betas[layer][3][
                                  0] * level12_new_3

                level24_new_1, level24_new_2 = self.cells[count](None,
                                                                 self.level_12[-1],
                                                                 self.level_24[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level24_new = normalized_betas[layer][2][2] * level24_new_1 + normalized_betas[layer][3][
                    1] * level24_new_2

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            else:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2, level6_new_3 = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             self.level_12[-1],
                                                                             normalized_alphas)
                count += 1

                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][
                    1] * level6_new_2 + normalized_betas[layer][2][
                                 0] * level6_new_3

                level12_new_1, level12_new_2, level12_new_3 = self.cells[count](self.level_12[-2],
                                                                                self.level_6[-1],
                                                                                self.level_12[-1],
                                                                                self.level_24[-1],
                                                                                normalized_alphas)
                count += 1
                level12_new = normalized_betas[layer][1][2] * level12_new_1 + normalized_betas[layer][2][
                    1] * level12_new_2 + normalized_betas[layer][3][
                                  0] * level12_new_3

                level24_new_1, level24_new_2 = self.cells[count](self.level_24[-2],
                                                                 self.level_12[-1],
                                                                 self.level_24[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level24_new = normalized_betas[layer][2][2] * level24_new_1 + normalized_betas[layer][3][
                    1] * level24_new_2

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            self.level_3 = self.level_3[-2:]
            self.level_6 = self.level_6[-2:]
            self.level_12 = self.level_12[-2:]
            self.level_24 = self.level_24[-2:]

        # define upsampling
        h, w = stem2.size()[2], stem2.size()[3]
        upsample_6 = nn.Upsample(size=stem2.size()[2:], mode='bilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[h // 2, w // 2], mode='bilinear', align_corners=True)
        upsample_24 = nn.Upsample(size=[h // 4, w // 4], mode='bilinear', align_corners=True)

        result_3 = self.last_3(self.level_3[-1])
        result_6 = self.last_3(upsample_6(self.last_6(self.level_6[-1])))
        result_12 = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(self.level_12[-1])))))
        result_24 = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(self.last_24(self.level_24[-1]))))))

        sum_feature_map = result_3 + result_6 + result_12 + result_24
        return sum_feature_map

        # --------------follow from SuperGlue matcher------------
        """a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']


        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)

        self._arch_parameters = [
            alphas,
            betas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in
         zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


