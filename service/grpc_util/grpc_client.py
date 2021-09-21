import grpc
from service.grpc_util.proto import message_pb2, message_pb2_grpc
from typing import List


class Client:
    """Client of Molecular Properties Prediction"""

    def __init__(self, ip='localhost', port='8888'):
        channel = grpc.insecure_channel('{}:{}'.format(ip, port))
        self.stub_ = message_pb2_grpc.MolePropServerStub(channel)

    def get_mole_prop(self, tasks: List[str] = None, smiles: List[str] = None):
        """
        Get molecular properties.
        :param tasks: molecular property prediction tasks.
        :param smiles: input data.
        :return:
        """
        response = self.stub_.get_mole_prop(message_pb2.Request(tasks=tasks, smiles=smiles))
        return response
