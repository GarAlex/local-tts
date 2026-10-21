//! Exposing an intermediate tensor of an ONNX model as an extra graph
//! output, in memory, with no `onnx` crate: a `ModelProto` is protobuf, its
//! graph is field 7, and a graph's outputs are its repeated field 12
//! (`ValueInfoProto`: name 1, type 2 → `TypeProto.tensor_type` 1 →
//! `elem_type` 1). Appending one output and re-encoding the graph's length
//! prefix is the whole change; every other byte is kept, so the file on
//! disk and its pinned SHA-256 never change.

/// ONNX `TensorProto.DataType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemType {
    Float = 1,
    Int64 = 7,
}

fn varint(bytes: &[u8], mut at: usize) -> Result<(u64, usize), String> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *bytes.get(at).ok_or("truncated ONNX varint")?;
        at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok((value, at));
        }
        shift += 7;
        if shift > 63 {
            return Err("invalid ONNX varint".into());
        }
    }
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_bytes(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    put_varint(out, field << 3 | 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// Every top-level field of a message: (field, wire type, start of the
/// whole field, start of its payload, end).
/// (field, wire type, field start, payload start, end).
type Field = (u64, u64, usize, usize, usize);

fn fields(bytes: &[u8]) -> Result<Vec<Field>, String> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let start = at;
        let (key, next) = varint(bytes, at)?;
        let (field, wire) = (key >> 3, key & 7);
        at = next;
        let payload = at;
        at = match wire {
            0 => varint(bytes, at)?.1,
            1 => at + 8,
            2 => {
                let (len, next) = varint(bytes, at)?;
                next + len as usize
            }
            5 => at + 4,
            _ => return Err(format!("unsupported ONNX wire type {wire}")),
        };
        if at > bytes.len() {
            return Err("truncated ONNX field".into());
        }
        let payload = if wire == 2 {
            varint(bytes, payload)?.1
        } else {
            payload
        };
        out.push((field, wire, start, payload, at));
    }
    Ok(out)
}

/// Graph output names, in order.
pub fn outputs(model: &[u8]) -> Result<Vec<String>, String> {
    let graph = graph_range(model)?;
    let bytes = &model[graph.0..graph.1];
    let mut names = Vec::new();
    for (field, _, _, payload, end) in fields(bytes)? {
        if field == 12 {
            for (f, _, _, p, e) in fields(&bytes[payload..end])? {
                if f == 1 {
                    names.push(String::from_utf8_lossy(&bytes[payload..end][p..e]).into_owned());
                }
            }
        }
    }
    Ok(names)
}

fn graph_range(model: &[u8]) -> Result<(usize, usize), String> {
    fields(model)?
        .into_iter()
        .find(|f| f.0 == 7 && f.1 == 2)
        .map(|f| (f.3, f.4))
        .ok_or_else(|| "ONNX model has no graph".into())
}

/// `model` with the tensor `name` added as a graph output called `as_name`
/// of element type `elem`. Already an output: unchanged (idempotent).
pub fn expose(model: &[u8], name: &str, as_name: &str, elem: ElemType) -> Result<Vec<u8>, String> {
    if outputs(model)?.iter().any(|o| o == as_name) {
        return Ok(model.to_vec());
    }
    let (start, end) = graph_range(model)?;
    // The output: an Identity node renames the tensor, then the new name is
    // declared an output — the graph's own value stays where it was.
    let mut node = Vec::new();
    put_bytes(&mut node, 1, name.as_bytes()); // input
    put_bytes(&mut node, 2, as_name.as_bytes()); // output
    put_bytes(&mut node, 3, format!("{as_name}_expose").as_bytes()); // name
    put_bytes(&mut node, 4, b"Identity"); // op_type
    let mut tensor = Vec::new();
    put_varint(&mut tensor, 1 << 3); // elem_type, varint
    put_varint(&mut tensor, elem as u64);
    let mut kind = Vec::new();
    put_bytes(&mut kind, 1, &tensor); // tensor_type
    let mut info = Vec::new();
    put_bytes(&mut info, 1, as_name.as_bytes());
    put_bytes(&mut info, 2, &kind);
    let mut graph = model[start..end].to_vec();
    put_bytes(&mut graph, 1, &node);
    put_bytes(&mut graph, 12, &info);
    let header_start = fields(model)?
        .into_iter()
        .find(|f| f.0 == 7 && f.1 == 2)
        .map(|f| f.2)
        .expect("graph found above");
    let mut out = Vec::with_capacity(model.len() + 128);
    out.extend_from_slice(&model[..header_start]);
    put_bytes(&mut out, 7, &graph);
    out.extend_from_slice(&model[end..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model of one field-7 graph holding one output, field 8 after it.
    fn tiny() -> Vec<u8> {
        let mut info = Vec::new();
        put_bytes(&mut info, 1, b"waveform");
        let mut graph = Vec::new();
        put_bytes(&mut graph, 12, &info);
        let mut model = Vec::new();
        put_varint(&mut model, 1 << 3);
        put_varint(&mut model, 8); // ir_version
        put_bytes(&mut model, 7, &graph);
        put_bytes(&mut model, 8, b"opset");
        model
    }

    #[test]
    fn expose_appends_one_output_and_keeps_every_other_byte() {
        let model = tiny();
        let patched = expose(
            &model,
            "/encoder/Clip_output_0",
            "durations",
            ElemType::Float,
        )
        .unwrap();
        assert_eq!(outputs(&patched).unwrap(), ["waveform", "durations"]);
        assert_eq!(&patched[..2], &model[..2], "fields before the graph");
        assert!(
            patched.ends_with(&model[model.len() - 7..]),
            "fields after the graph"
        );
        // Idempotent.
        assert_eq!(
            expose(
                &patched,
                "/encoder/Clip_output_0",
                "durations",
                ElemType::Float
            )
            .unwrap(),
            patched
        );
        assert!(expose(b"\xff", "x", "y", ElemType::Int64).is_err());
    }
}
