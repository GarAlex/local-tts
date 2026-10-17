# Dataset v1

## Beginner walkthrough in Studio

Open **Build dataset → First dataset: step-by-step guide**. The guide opens by
default, including before a dataset exists. Its six pages explain setup, truthful
rights/permission notes, choosing prompts, recording and reviewing each take,
held-out splits and validation, and the handoff to fine-tuning. Use **Back/Next**
or click a page title; navigation only changes help text. You can collapse the
guide at any time to give the editor more space.

The guide shows current saved clip counts by split, unsaved manifest changes and
an unadded take. These are editor facts, not a quality score or a completed-task
checklist. Start by practising the twelve bundled prompts, then collect a larger,
varied corpus and evaluate held-out speech. No fixed clip count guarantees a
usable voice.

## Manifest and editing

Store `dataset.json` beside an `audio` directory. Audio paths are relative to the manifest directory; parent traversal and symlinks outside this directory are rejected. Input is mono WAV at the declared sample rate. The checker does not automatically resample or alter recordings.

Studio's **Build dataset** tab creates this manifest. Import mono/stereo PCM WAV at 8–192 kHz (up to five minutes), or record the default microphone (up to twenty seconds). Imports are downmixed and converted with a windowed-sinc resampler to mono 22050 Hz. Trim each take to 0.5–20 seconds, enter its exact transcript, preview, then add it. Source files remain untouched. Recording requires your operating system's microphone permission; hardware capture has not yet been verified on this machine.

Use **Assign held-out splits** after adding clips, then **Save edits**. The deterministic seed groups identical normalized transcripts together, reserves approximately ten percent each for validation/test (at least one group each when three groups exist), and assigns the rest to training. Two distinct groups produce training and validation only. Review splits before training. Existing clips' transcripts and splits are editable. External manifest changes block saving until reopening (Discard manifest edits and reload provides recovery), and failed additions roll back their new WAV.

These operations are reusable through `local_tts_core::clips` and `local_tts_core::collection::Collection`; they require no GUI or Python.

```json
{
  "schema_version": 1,
  "language": "en",
  "sample_rate": 22050,
  "rights": {
    "audio": "Owned recordings; agreement reference ...",
    "transcripts": "Original scripts; source reference ...",
    "speaker_consent": "Permission for commercial synthetic voice use; agreement reference ..."
  },
  "utterances": [
    {
      "id": "speaker1-0001",
      "audio": "audio/speaker1-0001.wav",
      "text": "The train arrives at nine tomorrow morning.",
      "speaker": "speaker1",
      "split": "train"
    }
  ]
}
```

Languages: `en`, `ru`, `uk`. Splits: `train`, `validation`, `test`. Reserve held-out recordings before training. Audio aliases are rejected, while repeated normalized transcripts across splits produce warnings. Training preparation additionally rejects identical file contents at different paths. Clips outside 0.5–20 seconds and full-scale peaks produce warnings. Near-silent clips and decoding errors fail validation.

Rights fields must be populated, but the checker cannot determine whether their claims are true. Keep contracts separately; do not commit private personal information.

## Guided recording

In Build dataset, expand **Guided recording prompts**. Load starter prompts for
its language or import a UTF-8 text file with one sentence per line. Choose
**Use next unrecorded prompt**, record/import audio, check the exact words and trim,
then Add clip. Reopen the dataset and reload the same list to resume: matching
uses normalized saved transcripts for the selected speaker across all splits.
Editing a transcript may make its original prompt appear unrecorded again.

Lists allow at most one megabyte, ten thousand unique prompts and three hundred
characters per prompt. Blank lines are ignored; normalized duplicates and control
characters fail validation. Imported prompt lists stay in their original file;
keep that file to resume. Loading a list does not overwrite the current transcript;
Use next unrecorded prompt explicitly replaces it when no take is pending.

The bundled English, Russian and Ukrainian sentences are original project text
under MIT. Their twelve sentences per language help exercise recording; they are
not phonetic coverage sets or enough material for a usable voice. Broader corpora
and native-speaker review remain necessary. Custom prompts require appropriate
transcript rights, just as recordings require audio rights and speaker permission.

## Dataset statistics

Check saved dataset in Studio, Check dataset in the training tab, or CLI
`dataset-check` reports speaker/split statistics. Each group includes declared
utterances, error-free valid clips, duration and shortest/longest valid clips.
Invalid or duplicate-path entries do not contribute to group duration. Global
metadata errors still block training even when individual audio entries decode.
The legacy top-level duration counts readable WAV header duration; use the group
statistics when assessing validated recording time. Statistics do not detect
identical audio bytes at different paths; training preparation rejects those.
The initial trainer still requires exactly one speaker per voice.

## Russian/Ukrainian explicit stress

New preparations use `grapheme-v2`, shared by own-voice inference. You may mark
stress using `+` immediately before a vowel, or a Unicode combining acute accent
immediately after it: `з+амок` and `за́мок` produce identical tokens. Only one
explicit stress per word is supported. Malformed markers fail preparation.
Unicode NFC also normalizes decomposed letters; ё, ґ, є, і and ї stay distinct.
Prompt matching, split grouping and repeated-transcript checks canonicalize the
two marker spellings identically.

Annotate the pronunciation actually spoken in each recording, consistently
throughout the corpus. The model must see these annotations in training; a neutral
model cannot acquire reliable stress control just by adding a marker at inference.
There is no automatic stress dictionary, disambiguation or native-speaker quality
validation yet. Optional annotations need not be added to every word.

Legacy `grapheme-v1` voice packages retain their original frontend. Keep original
prepared configs when resuming older checkpoints; regenerating a v2 preparation
changes the dataset/config digest and correctly prevents resuming that old run.
New v2 runs should use a new checkpoint folder. Re-exporting legacy checkpoints
preserves v1 through the exporter's compatibility fallback.
