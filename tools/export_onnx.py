"""Export the model to ONNX and quantise it to int8 with per-token activation scales.

    python tools/export_onnx.py --model-dir _export/FRIDA-Decisions

Writes `<model-dir>/onnx/model_int8_pertoken.onnx` (plus `.onnx_data` when the
graph does not fit a single protobuf). The float32 graph is built in a
temporary directory and is not kept.

Graph inputs are the packed layout, so packing works exactly as in torch:

    input_ids  int64 (rows, L)
    buckets    uint8 (rows, L, L)   relative-position buckets
    allowed    bool  (rows, L, L)   block mask
    -> token_scores float32 (rows, L)   head score of every token

A candidate's margin is the mean of `token_scores` over its tokens.

Quantisation. Every weight MatMul of the encoder (q, k, v, o and the three
FFN projections) becomes `MatMulInteger` with int8 weights (one scale per
output channel) and uint8 activations quantised on the fly with ONE SCALE PER
TOKEN. A single scale per tensor is not enough here: a handful of activation
outliers would set the scale for every token and wash out the rest. uint8
activations with zero point 128 are used because onnxruntime's u8 x s8 kernel
is much faster than s8 x s8 on CPU. The decision head stays in float32.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frida_decisions.constants import ONNX_INT8_FILE, ONNX_SUBDIR  # noqa: E402

# Protobuf cannot hold more than 2 GiB; stay clear of it.
SINGLE_FILE_LIMIT = 1_900 * 2**20

SAMPLE_REQUEST = {
    "state": "Клиент пишет: списали деньги дважды за один заказ, прошу вернуть. " * 6,
    "questions": {
        "refund": {"type": "noul", "instructions": "Клиент просит вернуть деньги?"},
        "topic": {"type": "choice", "instructions": "Какая тема обращения?",
                  "criteria": {"billing": "оплата", "delivery": "доставка", "account": "аккаунт"}},
        "anger": {"type": "score", "instructions": "Насколько клиент раздражён?",
                  "criteria": ["спокоен", "раздражён", "в ярости"]},
    },
}


def export_fp32(model_dir: Path, path: Path):
    """Trace the float32 encoder + head to ONNX. Returns the sample inputs and
    the torch reference output for a later check."""
    import torch

    from frida_decisions.modeling import TokenScores
    from frida_decisions.packing import pack
    from frida_decisions.torch_backend import Judge

    judge = Judge.from_pretrained(model_dir, device="cpu", dtype=torch.float32, state_cache_mb=0)
    _, _, tok = judge.compile(SAMPLE_REQUEST)
    batch = pack([tok], judge.config)
    args = (torch.from_numpy(batch.input_ids), torch.from_numpy(batch.buckets),
            torch.from_numpy(batch.allowed))
    wrapper = TokenScores(judge.model).eval()
    with torch.inference_mode():
        reference = wrapper(*args).numpy()
    dynamic = {"input_ids": {0: "rows", 1: "length"},
               "buckets": {0: "rows", 1: "length", 2: "length"},
               "allowed": {0: "rows", 1: "length", 2: "length"},
               "token_scores": {0: "rows", 1: "length"}}
    with torch.no_grad():
        torch.onnx.export(wrapper, args, str(path), input_names=["input_ids", "buckets", "allowed"],
                          output_names=["token_scores"], dynamic_axes=dynamic,
                          opset_version=17, dynamo=False)
    inputs = {"input_ids": batch.input_ids, "buckets": batch.buckets, "allowed": batch.allowed}
    return inputs, reference, batch


def quantize_per_token(model):
    """Rewrite weight MatMuls to int8 x uint8 `MatMulInteger` with per-token scales."""
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    consts = [numpy_helper.from_array(np.float32(127.0), "q_c127"),
              numpy_helper.from_array(np.float32(-127.0), "q_cm127"),
              numpy_helper.from_array(np.float32(128.0), "q_c128"),
              numpy_helper.from_array(np.float32(1e-8), "q_eps"),
              numpy_helper.from_array(np.uint8(128), "q_zp128")]
    nodes, new_inits, replaced, count = [], [], set(), 0
    for node in graph.node:
        weight = inits.get(node.input[1]) if node.op_type == "MatMul" and len(node.input) > 1 else None
        if weight is None or len(weight.dims) != 2 or weight.dims[1] == 1:
            nodes.append(node)          # activations x activations, or the 1-wide head: keep
            continue
        w = numpy_helper.to_array(weight).astype(np.float32)               # (in, out)
        w_scale = np.maximum(np.abs(w).max(axis=0), 1e-8) / 127.0           # per output channel
        w_q = np.clip(np.round(w / w_scale), -127, 127).astype(np.int8)
        p = f"q{count}_"
        count += 1
        new_inits += [numpy_helper.from_array(w_q, p + "w"),
                      numpy_helper.from_array(w_scale.astype(np.float32), p + "w_scale")]
        x = node.input[0]

        def add(op, ins, out, **attrs):
            nodes.append(helper.make_node(op, ins, [out], name=p + out, **attrs))

        # per-token scale: max |x| over the hidden dimension / 127
        add("Abs", [x], p + "abs")
        add("ReduceMax", [p + "abs"], p + "amax", axes=[-1], keepdims=1)
        add("Div", [p + "amax", "q_c127"], p + "scale_raw")
        add("Max", [p + "scale_raw", "q_eps"], p + "scale")
        add("Div", [x, p + "scale"], p + "x_scaled")
        add("Round", [p + "x_scaled"], p + "x_round")
        add("Clip", [p + "x_round", "q_cm127", "q_c127"], p + "x_clip")
        add("Add", [p + "x_clip", "q_c128"], p + "x_shift")
        add("Cast", [p + "x_shift"], p + "x_u8", to=TensorProto.UINT8)
        add("MatMulInteger", [p + "x_u8", p + "w", "q_zp128"], p + "y_i32")
        add("Cast", [p + "y_i32"], p + "y_f32", to=TensorProto.FLOAT)
        add("Mul", [p + "y_f32", p + "scale"], p + "y_tok")
        nodes.append(helper.make_node("Mul", [p + "y_tok", p + "w_scale"], [node.output[0]],
                                      name=p + "out"))
        replaced.add(weight.name)
    still_used = {i for n in nodes for i in n.input}
    keep = [i for i in graph.initializer if i.name not in replaced or i.name in still_used]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(keep + consts + new_inits)
    return count


def scrub(model) -> None:
    """Remove doc strings and metadata that may carry local file paths."""
    model.doc_string = ""
    model.producer_name = "frida_decisions"
    model.producer_version = ""
    del model.metadata_props[:]
    for node in model.graph.node:
        node.doc_string = ""
        del node.metadata_props[:]
    model.graph.doc_string = ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, help="folder written by export_model.py")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tmp", default=None, help="where to build the float32 graph")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    import onnx
    import onnxruntime as ort

    model_dir = Path(args.model_dir)
    out_dir = model_dir / ONNX_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ONNX_INT8_FILE
    data_name = ONNX_INT8_FILE + "_data"

    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        fp32_path = Path(tmp) / "model_fp32.onnx"
        started = time.time()
        inputs, reference, batch = export_fp32(model_dir, fp32_path)
        gc.collect()
        print(f"float32 graph exported in {time.time() - started:.0f}s")

        model = onnx.load(str(fp32_path), load_external_data=True)
        n = quantize_per_token(model)
        scrub(model)
        print(f"quantised {n} MatMuls")
        for stale in (target, out_dir / data_name):
            if stale.exists():
                stale.unlink()
        size = sum(len(i.raw_data) for i in model.graph.initializer)
        if size < SINGLE_FILE_LIMIT:
            onnx.save(model, str(target))
        else:
            onnx.save(model, str(target), save_as_external_data=True, all_tensors_to_one_file=True,
                      location=data_name, size_threshold=1024)
        del model
        gc.collect()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
    scores = session.run(["token_scores"], inputs)[0]
    from frida_decisions.packing import mean_readout
    drift = np.abs(mean_readout(scores, batch.readout) - mean_readout(reference, batch.readout)).max()
    files = sorted(p for p in out_dir.iterdir())
    total = sum(p.stat().st_size for p in files)
    print(f"wrote {[p.name for p in files]} ({total / 2**20:.0f} MiB); "
          f"sample margin drift int8 vs float32: {drift:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
