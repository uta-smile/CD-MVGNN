from argparse import Namespace

from torch import nn as nn

from dglt.contrib.grover.models import GroverFinetuneTask
from dglt.models.nn_utils import initialize_weights
from dglt.models.zoo import MPNN
from dglt.models.zoo.dgcnn import DGCNN
from dglt.models.zoo.mpnn import MultiMPNN, DualMPNN, DualMPNNPlus


def build_finetune_model(model, args: Namespace, model_idx=0):
    """
    Build a model, which is a mpnn + fine tune fc layers.
    :param model: The loaded model to be fine tuned.
    :param args: Arguments.
    :param model_idx: The model index, used for distinct init.
    :return: The model.
    """
    model.set_finetune_layers(args)
    initialize_weights(model=model.ffn)
    return model


def build_model(args: Namespace, model_idx=0) -> nn.Module:
    """
    Builds a MPNN, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MPNN containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    # New
    if args.model_type == "dualmpnn":
        model = DualMPNN(args)
    elif args.model_type == "mpnn":
        model = MPNN(classification=args.dataset_type == 'classification')
        model.create_encoder(args)
        model.create_ffn(args)
    elif args.model_type == "multimpnn":
        model = MultiMPNN(args)
    elif args.model_type == '3dgcn':
        # model = TDGCN(args)
        model = DGCNN(args)
    elif args.model_type == "grover":
        model = GroverFinetuneTask(args)
    elif args.model_type == "dualmpnnplus":
        model = DualMPNNPlus(args)
    else:
        raise NotImplementedError(
            "%s is not implemented. Current available model: mpnn, multimpnn, dualmpnn and dualmpnnplus." % (args.model_type))
    initialize_weights(model=model, model_idx=model_idx, distinct_init=args.distinct_init)

    return model
