"""Compile NVIDIA NeMo's English text-normalization grammars and export them
for local-tts's native runner (crates/tts-core/src/nemo.rs).

Run inside the pinned container (training/nemo/build_grammars.sh). Writes to
the mounted /out:
  classify.fst.gz, verbalize.fst.gz, post.fst.gz  OpenFst binaries, gzipped
  manifest.json                                    versions and checksums
  reference.json                                   NeMo's English TN test cases
                                                   with Python NeMo's outputs —
                                                   the parity target
  LICENSE                                          NeMo's Apache-2.0 licence
"""
import glob, gzip, hashlib, io, json, os, shutil, tarfile, tempfile, urllib.request
from importlib.metadata import distribution, version

import pynini
from nemo_text_processing.text_normalization.normalize import Normalizer

TAG = "r1.2.0"  # the NeMo-text-processing release the pip package was cut from
OUT = "/out"
cache = tempfile.mkdtemp()
normalizer = Normalizer(input_case="cased", lang="en", deterministic=True, cache_dir=cache)

names = {
    "tokenize_and_classify": "classify",
    "verbalize": "verbalize",
    "post_process_graph": "post",
}
manifest = {
    "nemo_text_processing": version("nemo_text_processing"),
    "pynini": version("pynini"),
    "source": f"https://github.com/NVIDIA/NeMo-text-processing/tree/{TAG}",
    "config": {"lang": "en", "input_case": "cased", "deterministic": True},
    "files": {},
}
for far_path in sorted(glob.glob(f"{cache}/*.far")):
    far = pynini.Far(far_path, mode="r")
    while not far.done():
        key, fst = far.get_key(), far.get_fst()
        name = names[key]
        raw = f"{cache}/{name}.fst"
        fst.write(raw)
        data = open(raw, "rb").read()
        with open(f"{OUT}/{name}.fst.gz", "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0) as gz:
                gz.write(data)
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        far.next()

dist = distribution("nemo_text_processing")
licence = next(f for f in dist.files if f.name.upper().startswith("LICENSE"))
shutil.copy(dist.locate_file(licence), f"{OUT}/LICENSE")

raw = urllib.request.urlopen(
    f"https://github.com/NVIDIA/NeMo-text-processing/archive/refs/tags/{TAG}.tar.gz", timeout=300
).read()
cases = []
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    for m in sorted(tar.getmembers(), key=lambda m: m.name):
        if "/tests/nemo_text_processing/en/data_text_normalization/test_cases_" in m.name and m.isfile():
            kind = m.name.rsplit("test_cases_", 1)[1][:-4]
            if kind in ("normalize_with_audio", "punctuation_match_input"):
                continue  # other modes than the one exported here
            for line in tar.extractfile(m).read().decode().splitlines():
                if "~" in line:
                    written, expected = line.split("~", 1)
                    python = normalizer.normalize(written, punct_post_process=False)
                    cases.append({"kind": kind, "written": written, "expected": expected, "python": python})
json.dump(cases, open(f"{OUT}/reference.json", "w"), ensure_ascii=False, indent=0)
manifest["reference_cases"] = len(cases)
manifest["python_matches_expected"] = sum(c["python"] == c["expected"] for c in cases)
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
print(json.dumps(manifest, indent=2))
