# Continuous numeric voice rollout

Updated 2026-10-16. A decoder component alone is not a complete numeric voice.
Full-text support elsewhere in the application also does not establish support
in the compact arithmetic backend.

| Language | Current compact arithmetic path | Continuous sequence path |
|---|---|---|
| English | Pretrained decoder and cached numeric variants | New sequence architecture not yet ported |
| Russian | Cached variants and native sequence v6/v7 | Training, compression and decoder QAT available; speech errors remain |
| Ukrainian | Cached variants available | Expanded 512-utterance student/native int8 v10; 2.93 MB download, experimental |
| French | No integrated arithmetic frontend | Checked int8 decoder only; corpus/frontend/student pending |
| German | No integrated arithmetic frontend | Checked int8 decoder only; corpus/frontend/student pending |
| Spanish | No integrated arithmetic frontend | Checked int8 decoder only; corpus/frontend/student pending |
| Italian | No integrated arithmetic frontend | Checked int8 decoder only; corpus/frontend/student pending |
| Portuguese | No integrated arithmetic frontend | Checked int8 decoder only; corpus/frontend/student pending |
| Polish | No integrated arithmetic frontend | Checked int8 decoder only; corpus/frontend/student pending |
| Dutch | No integrated arithmetic frontend | Checked int8 decoder only; corpus/frontend/student pending |

Ukrainian is first, following the established Russian/Ukrainian priority. Next
European sequence work is French, German and Spanish, then Italian, Portuguese,
Polish and Dutch. Preserve source notices and recording/initialization rights
qualifications for every artifact; no universal commercial clearance is implied.

For each language, completion requires all of the following:

1. A literal numeric reading contract preserving signs, operators, fractional
   zeros and equation text; use the same normalization in library, CLI and UI.
2. Native whole-utterance phonemes checked against any compact token lookup.
3. Disjoint train/validation/test spoken utterances with pinned teacher source,
   checksummed full-utterance latents and token alignments.
4. Student training and float-decoder controls before compression. Keep int8
   when it fits the download budget; stronger compression needs evidence.
5. Native export, dynamic-length/speed parity, request/integrity/cancellation
   checks, a verified archive and measured download and installed sizes.
6. Paired teacher/student listening and language-appropriate numeric fidelity
   checks. ASR is diagnostic, especially where teacher controls misrecognize.
7. Optional decoder adaptation trials, compared against the unadapted candidate;
   lower reconstruction loss alone is insufficient to replace the candidate.

The QAT tool now supports Russian and Ukrainian and both packed 4-bit and
per-axis int8 weight simulation. The native sequence runtime still deliberately
accepts RU/UK only. Other languages must not be advertised merely because their
vocoder geometry happens to match.

Checked European decoder inventory: `target/math-language-rollout-v10.json`.
All seven existing component checksums and source notices were checked in this
rollout; each decoder is approximately 1.77 MB before adding a frontend/student.
