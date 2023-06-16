# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: eval_dispatcher.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name='eval_dispatcher.proto',
    package='',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x15\x65val_dispatcher.proto\x1a\x1bgoogle/protobuf/empty.proto\"T\n\x07\x45valJob\x12)\n!json_policy_specs_for_each_player\x18\x01 \x03(\t\x12\x1e\n\x16required_games_to_play\x18\x02 \x01(\x03\"q\n\rEvalJobResult\x12)\n!json_policy_specs_for_each_player\x18\x01 \x03(\t\x12\x1f\n\x17payoffs_for_each_player\x18\x02 \x03(\x02\x12\x14\n\x0cgames_played\x18\x03 \x01(\x03\"\"\n\x10\x45valConfirmation\x12\x0e\n\x06result\x18\x01 \x01(\x08\x32\x7f\n\x0e\x45valDispatcher\x12\x31\n\x0bTakeEvalJob\x12\x16.google.protobuf.Empty\x1a\x08.EvalJob\"\x00\x12:\n\x13SubmitEvalJobResult\x12\x0e.EvalJobResult\x1a\x11.EvalConfirmation\"\x00\x62\x06proto3'
    ,
    dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR, ])

_EVALJOB = _descriptor.Descriptor(
    name='EvalJob',
    full_name='EvalJob',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='json_policy_specs_for_each_player', full_name='EvalJob.json_policy_specs_for_each_player', index=0,
            number=1, type=9, cpp_type=9, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='required_games_to_play', full_name='EvalJob.required_games_to_play', index=1,
            number=2, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=54,
    serialized_end=138,
)

_EVALJOBRESULT = _descriptor.Descriptor(
    name='EvalJobResult',
    full_name='EvalJobResult',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='json_policy_specs_for_each_player', full_name='EvalJobResult.json_policy_specs_for_each_player',
            index=0,
            number=1, type=9, cpp_type=9, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='payoffs_for_each_player', full_name='EvalJobResult.payoffs_for_each_player', index=1,
            number=2, type=2, cpp_type=6, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='games_played', full_name='EvalJobResult.games_played', index=2,
            number=3, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=140,
    serialized_end=253,
)

_EVALCONFIRMATION = _descriptor.Descriptor(
    name='EvalConfirmation',
    full_name='EvalConfirmation',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='result', full_name='EvalConfirmation.result', index=0,
            number=1, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=255,
    serialized_end=289,
)

DESCRIPTOR.message_types_by_name['EvalJob'] = _EVALJOB
DESCRIPTOR.message_types_by_name['EvalJobResult'] = _EVALJOBRESULT
DESCRIPTOR.message_types_by_name['EvalConfirmation'] = _EVALCONFIRMATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

EvalJob = _reflection.GeneratedProtocolMessageType('EvalJob', (_message.Message,), {
    'DESCRIPTOR': _EVALJOB,
    '__module__': 'eval_dispatcher_pb2'
    # @@protoc_insertion_point(class_scope:EvalJob)
})
_sym_db.RegisterMessage(EvalJob)

EvalJobResult = _reflection.GeneratedProtocolMessageType('EvalJobResult', (_message.Message,), {
    'DESCRIPTOR': _EVALJOBRESULT,
    '__module__': 'eval_dispatcher_pb2'
    # @@protoc_insertion_point(class_scope:EvalJobResult)
})
_sym_db.RegisterMessage(EvalJobResult)

EvalConfirmation = _reflection.GeneratedProtocolMessageType('EvalConfirmation', (_message.Message,), {
    'DESCRIPTOR': _EVALCONFIRMATION,
    '__module__': 'eval_dispatcher_pb2'
    # @@protoc_insertion_point(class_scope:EvalConfirmation)
})
_sym_db.RegisterMessage(EvalConfirmation)

_EVALDISPATCHER = _descriptor.ServiceDescriptor(
    name='EvalDispatcher',
    full_name='EvalDispatcher',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=291,
    serialized_end=418,
    methods=[
        _descriptor.MethodDescriptor(
            name='TakeEvalJob',
            full_name='EvalDispatcher.TakeEvalJob',
            index=0,
            containing_service=None,
            input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
            output_type=_EVALJOB,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='SubmitEvalJobResult',
            full_name='EvalDispatcher.SubmitEvalJobResult',
            index=1,
            containing_service=None,
            input_type=_EVALJOBRESULT,
            output_type=_EVALCONFIRMATION,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ])
_sym_db.RegisterServiceDescriptor(_EVALDISPATCHER)

DESCRIPTOR.services_by_name['EvalDispatcher'] = _EVALDISPATCHER

# @@protoc_insertion_point(module_scope)
