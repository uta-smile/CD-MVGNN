from argparse import Namespace

import numpy as np
import torch
from torch import nn as nn

from dglt.models.layers import Readout
from dglt.models.modules import DualMPN
from dglt.models.nn_utils import get_activation_function


class MolEmbedding(nn.Module):
    def __init__(self, args):
        super(MolEmbedding, self).__init__()
        # TODO: Adhoc here !
        args.use_input_features = False
        self.encoders = DualMPN(args=args, atom_emb_output=True)

    def forward(self, graph_batch):

        output = self.encoders(graph_batch, None)
        return {"atom": output[0], 'bond': output[1]}


class AtomVocabPrediction(nn.Module):
    def __init__(self, args, vocab_size):
        """

        :param args:
        :param vocab_size:
        """
        super(AtomVocabPrediction, self).__init__()
        self.linear = nn.Linear(args.hidden_size * 100, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        # lin = self.linear(embeddings)
        # lin = self.logsoftmax(lin)
        return self.logsoftmax(self.linear(embeddings))


class FunctionalGroupPrediction(nn.Module):
    def __init__(self, args, fg_size):
        super(FunctionalGroupPrediction, self).__init__()
        first_linear_dim = args.hidden_size * 100
        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=args.hidden_size, attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out, args=args)
            first_linear_dim = first_linear_dim * args.attn_out
        else:
            self.readout = Readout(rtype="mean", hidden_size=args.hidden_size, args=args)

        self.linear_atom = nn.Linear(first_linear_dim, fg_size)
        self.linear_bond = nn.Linear(first_linear_dim, fg_size)

    def forward(self, embeddings, ascope):
        atom_embeddings = embeddings["atom"]
        bond_embeddings = embeddings["bond"]

        atom_embeddings = self.linear_atom(self.readout(atom_embeddings, ascope))
        bond_embeddings = self.linear_bond(self.readout(bond_embeddings, ascope))

        return (atom_embeddings, bond_embeddings)


class GroverTask(nn.Module):
    def __init__(self, args, molbert, vocab_size, fg_size):
        super(GroverTask, self).__init__()
        self.molbert = molbert
        self.av_task_atom = AtomVocabPrediction(args, vocab_size)
        self.av_task_bond = AtomVocabPrediction(args, vocab_size)
        self.fg_task_all = FunctionalGroupPrediction(args, fg_size)

    @staticmethod
    def get_loss_func(args):
        def loss_func(preds, targets, dist_coff=args.dist_coff):
            av_task_loss = nn.NLLLoss(ignore_index=0, reduction="mean")
            fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")
            # av_task_dist_loss = nn.KLDivLoss(reduction="mean")
            av_task_dist_loss = nn.MSELoss(reduction="mean")
            fg_task_dist_loss = nn.MSELoss(reduction="mean")
            sigmoid = nn.Sigmoid()

            av_atom_loss = av_task_loss(preds['av_task'][0], targets["av_task"])
            av_bond_loss = av_task_loss(preds['av_task'][1], targets["av_task"])
            av_loss = av_atom_loss + av_bond_loss

            fg_atom_loss = fg_task_loss(preds["fg_task"][0], targets["fg_task"])
            fg_bond_loss = fg_task_loss(preds["fg_task"][1], targets["fg_task"])
            fg_loss = fg_atom_loss + fg_bond_loss


            av_dist_loss = av_task_dist_loss(preds['av_task'][0], preds['av_task'][1])
            fg_dist_loss = fg_task_dist_loss(sigmoid(preds["fg_task"][0]), sigmoid(preds["fg_task"][1]))


            dist_loss = av_dist_loss + fg_dist_loss
            # print("%.4f %.4f %.4f %.4f %.4f %.4f"%(av_atom_loss,
            #                                       av_bond_loss,
            #                                       fg_atom_loss,
            #                                       fg_bond_loss,
            #                                       av_dist_loss,
            #                                       fg_dist_loss))
            # return av_loss + fg_loss + dist_coff * dist_loss
            overall_loss = av_loss + fg_loss + dist_coff * av_dist_loss + fg_dist_loss
            return overall_loss, av_loss, fg_loss, av_dist_loss, fg_dist_loss

        return loss_func

    def forward(self, graph_batch):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, _ = graph_batch
        a_scope = a_scope.data.cpu().numpy().tolist()

        embeddings = self.molbert(graph_batch)

        av_task_pred_atom = self.av_task_atom(embeddings["atom"])

        av_task_pred_bond = self.av_task_bond(embeddings["bond"])
        fg_task_pred_all = self.fg_task_all(embeddings, a_scope)
        return {"av_task": (av_task_pred_atom, av_task_pred_bond),
                "fg_task": fg_task_pred_all}


class GroverFinetuneTask(nn.Module):
    def __init__(self, args):
        super(GroverFinetuneTask, self).__init__()
        self.molbert = MolEmbedding(args)

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=args.hidden_size, attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out, args=args)
        else:
            self.readout = Readout(rtype="mean", hidden_size=args.hidden_size, args=args)

        self.mol_atom_ffn = self.create_ffn(args)
        self.mol_bond_ffn = self.create_ffn(args)
        self.ffn = nn.ModuleList()
        self.ffn.append(self.mol_atom_ffn)
        self.ffn.append(self.mol_bond_ffn)
        self.args = args
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 100

            if args.self_attention:
                first_linear_dim = first_linear_dim * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size * 100)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size * 100, args.ffn_hidden_size * 100),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size * 100, args.output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    @staticmethod
    def get_loss_func(args):
        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=args.dist_coff):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression':
                pred_loss = nn.MSELoss(reduction='none')
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # TODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # if DualMPNN is in eval mode.
                return pred_loss(preds, targets)

            # if DualMPNN is in train mode.
            dist_loss = nn.MSELoss(reduction='none')
            # dist_loss = nn.CosineSimilarity(dim=0)
            # print(pred_loss)
            # print(kl_loss(preds[0], preds[1]))
            pt = pred_loss(preds[1], targets)

            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            # TODO: dist constraint should be discussed, as well as the coff.
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

    def forward(self, batch, features_batch):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, _ = batch

        output = self.molbert(batch)
        # Share readout
        mol_edge_output = self.readout(output["bond"], a_scope)
        mol_atom_output = self.readout(output["atom"], a_scope)

        # Concat feature.
        # TODO: inconsistent behavior of args.use_input_features
        features_batch = torch.from_numpy(np.stack(features_batch)).float()
        if self.args.cuda:
            features_batch = features_batch.cuda()
        features_batch = features_batch.to(output["atom"])
        if len(features_batch.shape) == 1:
            features_batch = features_batch.view([1, features_batch.shape[0]])

        output["atom"] = torch.cat([mol_atom_output, features_batch], 1)
        output["bond"] = torch.cat([mol_edge_output, features_batch], 1)

        if self.training:
            atom_ffn_output = self.mol_atom_ffn(output["atom"])
            bond_ffn_output = self.mol_bond_ffn(output["bond"])
            return atom_ffn_output, bond_ffn_output
        else:
            # TODO: Not a good implantation. Should be controlled by encoders.
            atom_ffn_output = self.mol_atom_ffn(output["atom"])
            bond_ffn_output = self.mol_bond_ffn(output["bond"])
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            output = (atom_ffn_output + bond_ffn_output) / 2
        return output
