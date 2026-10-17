"""Compress a trusted native voice's stored weights without relearning speech.

Weights use packed blockwise 4/5-bit or per-axis int8 storage; operations stay float32.
This reduces disk size, not necessarily memory or latency. No quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort


def packed_four_bit(values, name, block_size=64):
    """Portable float32 ONNX unpacking of two unsigned nibbles per byte."""
    if type(block_size) is not int or block_size < 2 or block_size > 1024 or block_size % 2:
        raise ValueError("4-bit block size must be an even integer in 2..1024")
    flat = values.reshape(-1)
    padded = np.pad(flat, (0, (-flat.size) % block_size)).reshape(-1, block_size)
    scale = np.maximum(np.max(np.abs(padded), axis=1)/7,
                       np.finfo(np.float32).tiny).astype(np.float32)
    q = np.clip(np.rint(padded/scale[:,None]), -7, 7).astype(np.int8).reshape(-1) + 8
    packed = (q[0::2].astype(np.uint8) | (q[1::2].astype(np.uint8) << 4))
    initializers = []
    def constant(suffix, array):
        ident = name + '.packed4.' + suffix
        initializers.append(numpy_helper.from_array(np.asarray(array), ident))
        return ident
    stored = constant('bytes', packed)
    scales = constant('scales', scale[:,None])
    sixteen = constant('sixteen', np.float32(16))
    eight = constant('eight', np.float32(8))
    axes = constant('axes', np.array([1], dtype=np.int64))
    blocks = constant('blocks', np.array([-1, block_size], dtype=np.int64))
    flatten = constant('flatten', np.array([-1], dtype=np.int64))
    start = constant('start', np.array([0], dtype=np.int64))
    end = constant('end', np.array([flat.size], dtype=np.int64))
    shape = constant('shape', np.array(values.shape, dtype=np.int64))
    nodes = []
    def op(kind, inputs, suffix, **attrs):
        output = name + '.packed4.' + suffix
        nodes.append(helper.make_node(kind, inputs, [output], **attrs))
        return output
    f = op('Cast', [stored], 'float', to=TensorProto.FLOAT)
    hi = op('Floor', [op('Div', [f,sixteen], 'div')], 'hi')
    lo = op('Sub', [f,op('Mul', [hi,sixteen], 'shift')], 'lo')
    pairs = op('Concat', [op('Unsqueeze',[lo,axes],'lo_axis'),
                          op('Unsqueeze',[hi,axes],'hi_axis')], 'pairs', axis=1)
    signed = op('Sub', [op('Reshape',[pairs,blocks],'blocks_value'),eight], 'signed')
    decoded = op('Mul',[signed,scales],'decoded')
    cropped = op('Slice',[op('Reshape',[decoded,flatten],'flat'),start,end],'cropped')
    nodes.append(helper.make_node('Reshape',[cropped,shape],[name]))
    return initializers, nodes


def packed_five_bit(values, name, block_size=64):
    """Pack eight 5-bit signed values into five bytes; unpack with float32 ops."""
    if type(block_size) is not int or not 8<=block_size<=1024 or block_size%8:
        raise ValueError('5-bit block size must be a multiple of eight in 8..1024')
    flat=values.reshape(-1)
    padded=np.pad(flat,(0,(-flat.size)%block_size)).reshape(-1,block_size)
    scale=np.maximum(np.abs(padded).max(1)/15,np.finfo(np.float32).tiny).astype(np.float32)
    q=(np.clip(np.rint(padded/scale[:,None]),-15,15).astype(np.int32)+16).reshape(-1,8)
    packed=np.zeros((len(q),5),dtype=np.uint8)
    for i in range(8):
        byte,shift=divmod(i*5,8)
        value=q[:,i].astype(np.uint32)<<shift
        packed[:,byte]|=(value & 255).astype(np.uint8)
        if shift>3:packed[:,byte+1]|=(value>>8).astype(np.uint8)
    constants=[];nodes=[]
    def const(suffix,array):
        ident=name+'.packed5.'+suffix
        constants.append(numpy_helper.from_array(np.asarray(array),ident));return ident
    def op(kind,inputs,suffix,**attrs):
        out=name+'.packed5.'+suffix;nodes.append(helper.make_node(kind,inputs,[out],**attrs));return out
    stored=const('bytes',packed);scales=const('scales',scale[:,None])
    value=op('Cast',[stored],'float',to=TensorProto.FLOAT)
    columns=[name+f'.packed5.byte{i}' for i in range(5)]
    nodes.append(helper.make_node('Split',[value,const('splits',np.ones(5,dtype=np.int64))],columns,axis=1))
    unpacked=[]
    for i in range(8):
        byte,shift=divmod(i*5,8)
        part=op('Floor',[op('Div',[columns[byte],const(f'div{i}',np.float32(2**shift))],f'divided{i}')],f'floor{i}')
        if shift>3:part=op('Add',[part,op('Mul',[columns[byte+1],const(f'mul{i}',np.float32(2**(8-shift)))],f'upper{i}')],f'combined{i}')
        thirtytwo=const(f'base{i}',np.float32(32))
        high=op('Floor',[op('Div',[part,thirtytwo],f'quotient{i}')],f'high{i}')
        unpacked.append(op('Sub',[part,op('Mul',[high,thirtytwo],f'removed{i}')],f'low{i}'))
    joined=op('Concat',unpacked,'joined',axis=1)
    signed=op('Sub',[op('Reshape',[joined,const('blocks',np.array([-1,block_size],dtype=np.int64))],'blocks_value'),const('zero',np.float32(16))],'signed')
    decoded=op('Mul',[signed,scales],'decoded')
    cropped=op('Slice',[op('Reshape',[decoded,const('flat',np.array([-1],dtype=np.int64))],'flat_value'),const('start',np.array([0],dtype=np.int64)),const('end',np.array([flat.size],dtype=np.int64))],'cropped')
    nodes.append(helper.make_node('Reshape',[cropped,const('shape',np.array(values.shape,dtype=np.int64))],[name]))
    return constants,nodes


def compress(model, bits=8, block_size=64, int8_names=()):
    if bits not in (4,5,8):
        raise ValueError('bits must be 4, 5 or 8')
    if bits==4 and (type(block_size) is not int or not 2<=block_size<=1024 or block_size%2):raise ValueError('invalid 4-bit block size')
    if bits==5 and (type(block_size) is not int or not 8<=block_size<=1024 or block_size%8):raise ValueError('invalid 5-bit block size')
    int8_names=set(int8_names)
    eligible={t.name for t in model.graph.initializer if t.data_type==TensorProto.FLOAT and len(t.dims)>=2 and np.prod(t.dims)>=1024}
    if not int8_names <= eligible:raise ValueError("unknown or ineligible int8 weight override")
    prefixes = {name for node in model.graph.node for name in (*node.input, *node.output)}
    prefixes.update(t.name for t in model.graph.initializer)
    retained, dequant, count = [], [], 0
    for tensor in model.graph.initializer:
        values = numpy_helper.to_array(tensor)
        if tensor.data_type != TensorProto.FLOAT or values.size < 1024 or values.ndim < 2:
            retained.append(tensor)
            continue
        if not np.isfinite(values).all():
            raise ValueError('non-finite source weights')
        if bits in (4,5) and tensor.name not in int8_names:
            pack=packed_four_bit if bits==4 else packed_five_bit
            constants, nodes = pack(values, tensor.name, block_size)
            created = {t.name for t in constants} | {out for n in nodes[:-1] for out in n.output}
            if prefixes & created:
                raise ValueError('compression name collision')
            prefixes.update(created)
            retained.extend(constants)
            dequant.extend(nodes)
            count += 1
            continue
        names = [tensor.name + suffix for suffix in ('.stored_int8', '.scale', '.zero')]
        if any(name in prefixes for name in names):
            raise ValueError('compression name collision')
        prefixes.update(names)
        # Axis 0 is valid for every stored weight, including ConvTranspose.
        scale = np.maximum(np.max(np.abs(values), axis=tuple(range(1, values.ndim))) / 127,
                           np.finfo(np.float32).tiny).astype(np.float32)
        expanded = scale.reshape((-1,) + (1,) * (values.ndim - 1))
        quantized = np.clip(np.rint(values / expanded), -127, 127).astype(np.int8)
        retained.extend([numpy_helper.from_array(quantized, names[0]),
                         numpy_helper.from_array(scale, names[1]),
                         numpy_helper.from_array(np.zeros(scale.shape, dtype=np.int8), names[2])])
        dequant.append(helper.make_node('DequantizeLinear', names, [tensor.name], axis=0))
        count += 1
    nodes = dequant + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    onnx.checker.check_model(model)
    return count


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bits', type=int, choices=[4,5,8], default=8)
    parser.add_argument('--block-size', type=int, default=64, help='quantization block size: even for 4-bit, multiple of eight for 5-bit')
    parser.add_argument('--arithmetic-dictionary', help='restrict pronunciation vocabulary to this checked source subset')
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists():
        parser.error('output already exists')
    checksums = json.loads((source / 'checksums.json').read_text())
    for name, expected in checksums.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != expected:
            raise ValueError('source package checksum mismatch')
    manifest = json.loads((source / 'voice.json').read_text())
    if manifest['model'] != 'model.onnx':
        raise ValueError('requires standard single-file ONNX export')
    model = onnx.load(source / 'model.onnx')
    count = compress(model, args.bits, args.block_size)
    shutil.copytree(source, output)
    if args.arithmetic_dictionary:
        subset = Path(args.arithmetic_dictionary).read_text()
        original_lines = set((source/'cmudict.dict').read_text().splitlines())
        lines = [line for line in subset.splitlines() if line and not line.startswith(';;;')]
        if not lines or any(line not in original_lines for line in lines):
            raise ValueError('arithmetic pronunciations must be an exact source subset')
        (output/'cmudict.dict').write_text(subset)
    onnx.save(model, output / 'model.onnx')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    original = ort.InferenceSession(str(source / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    compressed = ort.InferenceSession(str(output / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    references = json.loads((source / 'reference.json').read_text())
    errors = []
    for case in references:
        inputs = dict(tokens=np.array([case['tokens']], dtype=np.int64),
                      lengths=np.array([len(case['tokens'])], dtype=np.int64),
                      duration_scale=np.array([case['duration_scale']], dtype=np.float32))
        before = original.run(None, inputs)[0].reshape(-1)
        after = compressed.run(None, inputs)[0].reshape(-1)
        if not np.isfinite(after).all() or not len(after):
            raise ValueError('invalid compressed audio')
        n = min(len(before), len(after))
        errors.append(dict(original_samples=len(before), compressed_samples=len(after),
                           overlapping_rmse=float(np.sqrt(np.mean((before[:n]-after[:n])**2)))))
        case['samples'] = after.tolist()
    # New fixtures test native execution of the compressed graph. They do not
    # establish equivalence to the original; that comparison is recorded above.
    (output / 'reference.json').write_text(json.dumps(references))
    manifest['name'] += f'-stored-int{args.bits}-experimental'
    manifest['provenance'] += f'; experimental int{args.bits} stored weights, float32 execution'
    (output / 'voice.json').write_text(json.dumps(manifest, indent=2))
    report = dict(source_model_sha256=sha(source/'model.onnx'), compressed_tensors=count,
                  bits=args.bits, block_size=args.block_size if args.bits in (4,5) else None,
                  arithmetic_dictionary=bool(args.arithmetic_dictionary),
                  source_bytes=(source/'model.onnx').stat().st_size,
                  compressed_bytes=(output/'model.onnx').stat().st_size,
                  fixture_comparison=errors,
                  acceptance='pending listening and numeric accuracy; not distillation',
                  reference_origin='compressed ONNX Runtime CPU output; original differences separately recorded')
    (output/'compression.json').write_text(json.dumps(report, indent=2))
    checksums['compression.json'] = sha(output/'compression.json')
    (output/'checksums.json').write_text(json.dumps({name: sha(output/name) for name in checksums}, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
