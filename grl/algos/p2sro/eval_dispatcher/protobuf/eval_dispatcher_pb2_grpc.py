# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

import eval_dispatcher_pb2 as eval__dispatcher__pb2


class EvalDispatcherStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.TakeEvalJob = channel.unary_unary(
            '/EvalDispatcher/TakeEvalJob',
            request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            response_deserializer=eval__dispatcher__pb2.EvalJob.FromString,
        )
        self.SubmitEvalJobResult = channel.unary_unary(
            '/EvalDispatcher/SubmitEvalJobResult',
            request_serializer=eval__dispatcher__pb2.EvalJobResult.SerializeToString,
            response_deserializer=eval__dispatcher__pb2.EvalConfirmation.FromString,
        )


class EvalDispatcherServicer(object):
    """Missing associated documentation comment in .proto file."""

    def TakeEvalJob(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitEvalJobResult(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EvalDispatcherServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'TakeEvalJob': grpc.unary_unary_rpc_method_handler(
            servicer.TakeEvalJob,
            request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            response_serializer=eval__dispatcher__pb2.EvalJob.SerializeToString,
        ),
        'SubmitEvalJobResult': grpc.unary_unary_rpc_method_handler(
            servicer.SubmitEvalJobResult,
            request_deserializer=eval__dispatcher__pb2.EvalJobResult.FromString,
            response_serializer=eval__dispatcher__pb2.EvalConfirmation.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'EvalDispatcher', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class EvalDispatcher(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def TakeEvalJob(request,
                    target,
                    options=(),
                    channel_credentials=None,
                    call_credentials=None,
                    insecure=False,
                    compression=None,
                    wait_for_ready=None,
                    timeout=None,
                    metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EvalDispatcher/TakeEvalJob',
                                             google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                                             eval__dispatcher__pb2.EvalJob.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitEvalJobResult(request,
                            target,
                            options=(),
                            channel_credentials=None,
                            call_credentials=None,
                            insecure=False,
                            compression=None,
                            wait_for_ready=None,
                            timeout=None,
                            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EvalDispatcher/SubmitEvalJobResult',
                                             eval__dispatcher__pb2.EvalJobResult.SerializeToString,
                                             eval__dispatcher__pb2.EvalConfirmation.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
