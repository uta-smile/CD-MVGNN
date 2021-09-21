from argparse import Namespace

from torch import nn as nn

from dglt.models.modules import MPN, DualMPN, DualMPNPlus
from dglt.models.nn_utils import get_activation_function


class MPNN(nn.Module):
    """A MPNN is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool):
        """
        Initializes the MPNN.

        :param classification: Whether the model is a classification model.
        """
        super(MPNN, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = int(args.hidden_size * 100)
            if args.self_attention:
                first_linear_dim = first_linear_dim * args.attn_out

            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

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
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MPNN on input.

        :param input: Input.
        :return: The output of the MPNN.
        """
        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)

        return output


class MultiMPNN(nn.Module):
    def __init__(self, args):
        super(MultiMPNN, self).__init__()
        self.encoders = nn.ModuleList()

        if args.nencoders == 1:
            encoder = MPN(args=args)
            self.encoders.append(encoder)
        else:
            input_encoder = MPN(args=args, atom_emb_output=True)
            self.encoders.append(input_encoder)
            for i in range(args.nencoders - 2):
                mid_encoder = MPN(args=args,
                                  bond_fdim=input_encoder.bond_fdim,
                                  atom_fdim=int(args.hidden_size * 100),
                                  atom_emb_output=True)
                self.encoders.append(mid_encoder)

            # hard coding here
            out_encoders = MPN(args=args,
                               bond_fdim=input_encoder.bond_fdim,
                               atom_fdim=int(args.hidden_size * 100),
                               atom_emb_output=False)
            self.encoders.append(out_encoders)

        self.ffn = self.create_ffn(args)

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
            first_linear_dim = int(args.hidden_size * 100)
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    def forward(self, batch, features_batch, adjs_batch):
        # TODO: Dense/Inception can apply here.
        for encoder in self.encoders:
            n_f_atom = encoder(batch, features_batch)
            # batch.set_new_atom_feature(n_f_atom)
            batch = [n_f_atom] + list(batch[1:])
        mol_output = n_f_atom
        output = self.ffn(mol_output)
        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)

        return output


class DualMPNN(nn.Module):
    def __init__(self, args):
        super(DualMPNN, self).__init__()
        self.encoders = DualMPN(args=args)

        self.mol_atom_ffn = self.create_ffn(args)
        self.mol_bond_ffn = self.create_ffn(args)
        self.ffn = nn.ModuleList()
        self.ffn.append(self.mol_atom_ffn)
        self.ffn.append(self.mol_bond_ffn)

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
            first_linear_dim = int(args.hidden_size * 100)

            if args.self_attention:
                first_linear_dim = first_linear_dim * args.attn_out

            if args.use_input_features:
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

    def get_loss_func(self, args):
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
        output = self.encoders(batch, features_batch)
        if self.training:
            atom_ffn_output = self.mol_atom_ffn(output[0])
            bond_ffn_output = self.mol_bond_ffn(output[1])
            return atom_ffn_output, bond_ffn_output
        else:
            # TODO: Not a good implantation. Should be controlled by encoders.
            atom_ffn_output = self.mol_atom_ffn(output[0])
            bond_ffn_output = self.mol_bond_ffn(output[1])
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            output = (atom_ffn_output + bond_ffn_output) / 2
        return output



class DualMPNNPlus(nn.Module):
    """
    A dualMPNN model using the cross-dependent message passing.

    """
    def __init__(self, args):
        super(DualMPNNPlus, self).__init__()
        self.encoders = DualMPNPlus(args=args)

        self.mol_atom_ffn = self.create_ffn(args)
        self.mol_bond_ffn = self.create_ffn(args)
        self.ffn = nn.ModuleList()
        self.ffn.append(self.mol_atom_ffn)
        self.ffn.append(self.mol_bond_ffn)

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
            first_linear_dim = int(args.hidden_size * 100)

            if args.self_attention:
                first_linear_dim = first_linear_dim * args.attn_out

            if args.use_input_features:
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

    def get_loss_func(self, args):
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
            pt = pred_loss(preds[1], targets)

            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            # TODO: dist constraint should be discussed, as well as the coff.
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

    def forward(self, batch, features_batch):
        output = self.encoders(batch, features_batch)
        if self.training:
            atom_ffn_output = self.mol_atom_ffn(output[0])
            bond_ffn_output = self.mol_bond_ffn(output[1])
            return atom_ffn_output, bond_ffn_output
        else:
            # TODO: Not a good implantation. Should be controlled by encoders.
            atom_ffn_output = self.mol_atom_ffn(output[0])
            bond_ffn_output = self.mol_bond_ffn(output[1])
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            output = (atom_ffn_output + bond_ffn_output) / 2
        return output
