# Russian training-base investigation

The [Vosk Russian publisher](https://huggingface.co/alphacep/vosk-tts-ru-multi)
publishes training weights in addition to the ONNX inference package. At revision
`0960eb904b9035a9fdbba2882933a4f5a84c525f`, the inspected `G_1000.pth` is
167,905,738 bytes with SHA-256
`8c525a91f0c8b0ff7dbbb40764724a68d6a3627cad9d04339cc66490f4b054e2`.
Restricted `weights_only=True` loading found 818 generator tensors and all five
speaker embeddings. The separate `vosk-ru-multi-v1` importer now supports this checkpoint; see [Russian base](PUBLIC_RU_BASE.md).

The source config requires a speaker-conditioned text encoder, transformer
coupling flows, mel posterior input, duration adversarial modules and a multiband
iSTFT vocoder. These differ from the currently supported project/LJSpeech/Piper
VITS profiles. Passing it to the ordinary single-speaker importer or dropping
unmatched tensors would change the model. The implemented adapter retains the complete architecture and strictly loads every tensor.

These steps are implemented with synthetic engineering fixtures: pinned importer,
mel/conditioning training, shared native dictionary/phonemes, subband/duration
losses, resume and native export. The publisher ONNX has a separate inverse-STFT
gain discrepancy; our normalized export matches PyTorch and the original
unnormalized conversion is verified separately against the publisher graph. The dictionary/data rights chain still needs resolution separately
from the publisher's Apache-2.0 model declaration and source license. Details of
the existing inference package and voice attributions remain in CYRILLIC.md.

Inspection artifacts: `target/base-discovery/vosk-ru-api.json`,
`target/base-discovery/vosk-ru/tensor-shapes.json`; pinned source files now live in
`voices/vits-vosk-ru-base/source`. The compatible base is installed at
`voices/vits-vosk-ru-base/base.pt`.
