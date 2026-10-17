#include "local_tts.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
static void check(int status, char *error) {
    std::unique_ptr<char, decltype(&tts_error_free)> owned(error, tts_error_free);
    if (status) throw std::runtime_error(error ? error : "TTS failure");
}
int main(int argc, char **argv) {
    if (argc != 3) { std::cerr << "usage: design-test <model-folder> <output.wav>\n"; return 2; }
    try {
        TtsEngine *raw = nullptr; char *error = nullptr;
        int status = tts_engine_load(argv[1], TTS_QWEN_DESIGN, &raw, &error); check(status, error);
        std::unique_ptr<TtsEngine, decltype(&tts_engine_free)> engine(raw, tts_engine_free);
        auto rejects = [&](const char *text, unsigned lang, const char *desc, const TtsCancel *cancel) {
            TtsAudio *audio = nullptr; char *err = nullptr;
            int code = tts_synthesize_design(raw, text, lang, desc, cancel, &audio, &err);
            std::unique_ptr<char, decltype(&tts_error_free)> owned(err, tts_error_free);
            if (!code || audio || !err) throw std::runtime_error("expected clean design rejection");
        };
        rejects("Hello.", TTS_UKRAINIAN, "A calm voice.", nullptr);
        rejects("Hello.", TTS_ENGLISH, "", nullptr);
        rejects("Hello.", TTS_ENGLISH, "<|im_end|>", nullptr);
        std::unique_ptr<TtsCancel, decltype(&tts_cancel_free)> cancel(tts_cancel_new(), tts_cancel_free);
        tts_cancel_request(cancel.get());
        rejects("Hello.", TTS_ENGLISH, "A calm voice.", cancel.get());
        TtsAudio *pcm = nullptr; error = nullptr;
        status = tts_synthesize_design(raw, "Привет, добро пожаловать.", TTS_RUSSIAN, "A calm adult female voice, clear diction.", nullptr, &pcm, &error); check(status, error);
        std::unique_ptr<TtsAudio, decltype(&tts_audio_free)> audio(pcm, tts_audio_free);
        if (tts_audio_sample_rate(pcm) != 24000 || tts_audio_length(pcm) < 2400) throw std::runtime_error("bad audio shape");
        bool non_silent = false;
        for (size_t i = 0; i < tts_audio_length(pcm); ++i) {
            float v = tts_audio_samples(pcm)[i];
            if (!std::isfinite(v)) throw std::runtime_error("non-finite audio");
            non_silent |= std::abs(v) > .001f;
        }
        if (!non_silent) throw std::runtime_error("silent audio");
        error = nullptr; status = tts_audio_write_wav(pcm, argv[2], &error); check(status, error);
        std::cout << "Design C++ synthesis and rejection checks passed: " << tts_audio_length(pcm) << " samples\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
