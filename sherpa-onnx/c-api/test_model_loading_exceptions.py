#!/usr/bin/env python3
"""生成极小 ONNX fixture 并用独立进程测试；只依赖 Python 标准库。

这些 Identity 图只验证加载与资源生命周期，不代表真实 ASR 推理验证。
用法：python3 test-model-loading-exceptions.py TEST_BINARY BUILD_TMP_DIR [REPEATS]
"""
import json
import pathlib
import subprocess
import sys


def varint(n):
    result = bytearray()
    while n > 127:
        result.append((n & 127) | 128)
        n >>= 7
    return bytes(result) + bytes([n])


def integer(field, n):
    return varint(field * 8) + varint(n)


def blob(field, value):
    value = value.encode() if isinstance(value, str) else value
    return varint(field * 8 + 2) + varint(len(value)) + value


def value_info(name, shape, dtype=1):
    dims = b"".join(blob(1, integer(1, dim)) for dim in shape)
    tensor = integer(1, dtype) + blob(2, dims)
    return blob(1, name) + blob(2, blob(1, tensor))


def model(inputs, outputs):
    # inputs: (name, shape, type); outputs: (source, name)。
    types = {name: (shape, kind) for name, shape, kind in inputs}
    graph = blob(2, "load-test")
    for source, dest in outputs:
        node = blob(1, source) + blob(2, dest) + blob(4, "Identity")
        graph += blob(1, node)
    for name, shape, kind in inputs:
        graph += blob(11, value_info(name, shape, kind))
    for source, dest in outputs:
        shape, kind = types[source]
        graph += blob(12, value_info(dest, shape, kind))
    return integer(1, 8) + blob(7, graph) + blob(8, integer(2, 13))


def fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "corrupt.onnx").write_bytes(b"not an ONNX model\n")
    (root / "tokens.txt").write_text("<blk> 0\na 1\n", encoding="utf-8")
    for name in ("tokenizer", "bad-tokenizer"):
        d = root / name
        d.mkdir(exist_ok=True)
        vocab = {"<|audio_pad|>": 0, "<asr_text>": 1, "<unk>": 2}
        (d / "vocab.json").write_text(json.dumps(vocab) if name == "tokenizer" else "broken json")
        (d / "merges.txt").write_text("#version: 0.2\n")
        (d / "tokenizer_config.json").write_text("{}")
    graphs = {
        "conv": ([('input', [1, 1, 1], 1)], [('input', 'output')]),
        "wrong": ([('x', [1], 1), ('extra', [1], 1)], [('x', 'output')]),
        "encoder": ([('input', [1, 1, 1], 1), ('mask', [1], 7)], [('input', 'output')]),
        "decoder": ([('ids', [1, 1], 7), ('audio', [1, 1, 1], 1),
                     ('mask', [1, 32], 7), ('position', [1], 7),
                     ('key', [1, 32, 1, 1], 1), ('value', [1, 32, 1, 1], 1)],
                    [('audio', 'logits'), ('key', 'new_key'), ('value', 'new_value')]),
        "vad": ([('input', [1, 576], 1), ('state', [2, 1, 128], 1), ('sr', [1], 7)],
                [('input', 'output'), ('state', 'stateN')]),
    }
    for name, (inputs, outputs) in graphs.items():
        (root / (name + '.onnx')).write_bytes(model(inputs, outputs))


def main():
    binary, root = sys.argv[1:3]
    repeats = sys.argv[3] if len(sys.argv) > 3 else "3"
    fixtures(pathlib.Path(root))
    for case in ('online', 'qwen', 'shared', 'structure', 'tokenizer', 'vad', 'embedding', 'punctuation'):
        result = subprocess.run([binary, case, root, repeats], capture_output=True, text=True, timeout=120)
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit(f'{case}: child exit {result.returncode}')
        # 确保测试实际进入加载失败路径，而非在文件存在性检查时提前返回。
        if 'failed:' not in result.stderr:
            raise SystemExit(f'{case}: missing construction exception diagnostic')
        print(result.stdout.strip())


if __name__ == '__main__':
    main()
