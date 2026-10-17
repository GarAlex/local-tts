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
    if (argc != 4) { std::cerr << "usage: math-test <model-folder> <8> <output.wav>\n"; return 2; }
    try {
        TtsEngine *raw = nullptr; char *error = nullptr;
        int status = tts_engine_load(argv[1], static_cast<uint32_t>(std::stoul(argv[2])), &raw, &error); check(status, error);
        std::unique_ptr<TtsEngine, decltype(&tts_engine_free)> engine(raw, tts_engine_free);
        TtsAudio *pcm = nullptr; error = nullptr;
        status = tts_synthesize(engine.get(), "12 + 225 = 237", TTS_ENGLISH, "default", 1, &pcm, &error); check(status, error);
        std::unique_ptr<TtsAudio, decltype(&tts_audio_free)> audio(pcm, tts_audio_free);
        if (tts_audio_sample_rate(pcm) != 16000 || tts_audio_length(pcm) < 2400) throw std::runtime_error("bad audio shape");
        bool non_silent = false;
        for (size_t i = 0; i < tts_audio_length(pcm); ++i) {
            float v = tts_audio_samples(pcm)[i];
            if (!std::isfinite(v)) throw std::runtime_error("non-finite audio");
            non_silent |= std::abs(v) > .001f;
        }
        if (!non_silent) throw std::runtime_error("silent audio");
        error = nullptr; status = tts_audio_write_wav(pcm, argv[3], &error); check(status, error);
        std::cout << "Math C++ synthesis passed: " << tts_audio_length(pcm) << " samples\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
