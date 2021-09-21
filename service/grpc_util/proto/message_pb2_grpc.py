# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from service.grpc_util.proto import message_pb2 as message__pb2


class MolePropServerStub(object):
    # missing associated documentation comment in .proto file
    pass

    def __init__(self, channel):
        """Constructor.

        Args:
          channel: A grpc.Channel.
        """
        self.get_mole_prop = channel.unary_unary(
            '/MolePropServer/get_mole_prop',
            request_serializer=message__pb2.Request.SerializeToString,
            response_deserializer=message__pb2.Response.FromString,
        )


class MolePropServerServicer(object):
    # missing associated documentation comment in .proto file
    pass

    def get_mole_prop(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MolePropServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'get_mole_prop': grpc.unary_unary_rpc_method_handler(
            servicer.get_mole_prop,
            request_deserializer=message__pb2.Request.FromString,
            response_serializer=message__pb2.Response.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'MolePropServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))