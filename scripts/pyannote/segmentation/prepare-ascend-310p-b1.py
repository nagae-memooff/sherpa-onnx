#!/usr/bin/env python3

"""Prepare the production Pyannote segmentation model for Ascend 310P ATC.

The CANN 9.1 compiler needs a static batch-1 graph, 4-D InstanceNormalization
inputs, and constant LSTM initial states. These rewrites are mathematically
equivalent for the supported [1, 1, 160000] input.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


INPUT_SHAPE = (1, 1, 160000)
OUTPUT_SHAPE = (1, 589, 7)
LSTM_STATE_SHAPE = (2, 1, 128)


def get_shape(value_info):
    ans = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            ans.append(dim.dim_value)
        else:
            ans.append(dim.dim_param)
    return tuple(ans)


def get_subgraph_inputs(node):
    ans = set()
    for attribute in node.attribute:
        graphs = []
        if attribute.type == onnx.AttributeProto.GRAPH:
            graphs.append(attribute.g)
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            graphs.extend(attribute.graphs)
        for graph in graphs:
            for child in graph.node:
                ans.update(value for value in child.input if value)
                ans.update(get_subgraph_inputs(child))
    return ans


def prune_unused_graph(model):
    required = {value.name for value in model.graph.output}
    kept_reversed = []
    for node in reversed(model.graph.node):
        if any(output in required for output in node.output):
            kept_reversed.append(node)
            required.update(value for value in node.input if value)
            required.update(get_subgraph_inputs(node))

    del model.graph.node[:]
    model.graph.node.extend(reversed(kept_reversed))

    kept_initializers = [
        value for value in model.graph.initializer if value.name in required
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def prepare(source: Path, output: Path):
    model = onnx.load(source)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
        raise ValueError("Expected one model input named 'x'")

    for dim, value in zip(
        model.graph.input[0].type.tensor_type.shape.dim, INPUT_SHAPE
    ):
        dim.ClearField("dim_param")
        dim.dim_value = value

    existing_names = {value.name for value in model.graph.initializer}
    state_name = "pyannote_lstm_zero_state_b1"
    axes_name = "pyannote_instance_norm_extra_axis"
    if state_name in existing_names or axes_name in existing_names:
        raise ValueError("Model already contains an Ascend preprocessing tensor")

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.zeros(LSTM_STATE_SHAPE, dtype=np.float32), state_name
            ),
            numpy_helper.from_array(
                np.asarray([2], dtype=np.int64), axes_name
            ),
        ]
    )

    lstm_count = 0
    instance_norm_count = 0
    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "LSTM":
            if len(node.input) < 7:
                raise ValueError(f"Unexpected LSTM input list: {node.name}")
            node.input[5] = state_name
            node.input[6] = state_name
            lstm_count += 1

        if node.op_type == "InstanceNormalization":
            original_input = node.input[0]
            original_output = node.output[0]
            input_4d = original_input + "_ascend_4d"
            output_4d = original_output + "_ascend_4d"
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    [original_input, axes_name],
                    [input_4d],
                    name=node.name + "_AscendUnsqueeze",
                )
            )
            node.input[0] = input_4d
            node.output[0] = output_4d
            new_nodes.append(node)
            new_nodes.append(
                helper.make_node(
                    "Squeeze",
                    [output_4d, axes_name],
                    [original_output],
                    name=node.name + "_AscendSqueeze",
                )
            )
            instance_norm_count += 1
        else:
            new_nodes.append(node)

    if lstm_count != 4 or instance_norm_count != 4:
        raise ValueError(
            "Expected 4 LSTM and 4 InstanceNormalization nodes; "
            f"given {lstm_count} and {instance_norm_count}"
        )

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    prune_unused_graph(model)

    used_domains = {node.domain for node in model.graph.node}
    kept_opsets = [
        item for item in model.opset_import if item.domain in used_domains
    ]
    del model.opset_import[:]
    model.opset_import.extend(kept_opsets)

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    if get_shape(model.graph.input[0]) != INPUT_SHAPE:
        raise ValueError(
            f"Unexpected prepared input: {get_shape(model.graph.input[0])}"
        )
    if get_shape(model.graph.output[0]) != OUTPUT_SHAPE:
        raise ValueError(
            f"Unexpected prepared output: {get_shape(model.graph.output[0])}"
        )

    onnx.save(model, output)
    print(f"Saved Ascend ATC input model to {output}")
    print(f"input_shape={INPUT_SHAPE}")
    print(f"output_shape={OUTPUT_SHAPE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Original Pyannote ONNX model")
    parser.add_argument("output", type=Path, help="Prepared batch-1 ONNX model")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite the original ONNX model")
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
