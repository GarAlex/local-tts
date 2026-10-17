#include "local_tts.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

static void check(int status, char *error) {
    std::unique_ptr<char, decltype(&tts_error_free)> owned(error, tts_error_free);
    if (status) throw std::runtime_error(error ? error : "TTS failed");
}
int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: european-test <chatterbox-folder> <output-directory> [reference.wav]\n";
        return 2;
    }
    try {
        TtsEngine *raw = nullptr;
        char *error = nullptr;
        int status = tts_engine_load_with_reference(argv[1], TTS_CHATTERBOX, argc == 4 ? argv[3] : nullptr, &raw, &error);
        check(status, error);
        std::unique_ptr<TtsEngine, decltype(&tts_engine_free)> engine(raw, tts_engine_free);
        error = nullptr;
        status = tts_engine_set_exaggeration(engine.get(), 0.65f, &error);
        check(status, error);
        error = nullptr;
        const int invalid = tts_engine_set_exaggeration(engine.get(), 2.0f, &error);
        if (invalid != 1 || !error) throw std::runtime_error("invalid expressiveness accepted");
        tts_error_free(error);
        struct Sample { uint32_t language; const char *code; const char *text; };
        const Sample samples[] = {
            {TTS_FRENCH, "fr", "Bonjour, le soleil entre par la fenêtre."},
            {TTS_GERMAN, "de", "Grüße aus Köln. Die Straße führt über eine Brücke."},
            {TTS_SPANISH, "es", "¡Buenos días! El niño está cerca del río."},
        };
        std::filesystem::create_directories(argv[2]);
        for (const auto &sample : samples) {
            error = nullptr;
            TtsAudio *raw_audio = nullptr;
            status = tts_synthesize(engine.get(), sample.text, sample.language, "default", 1.0f, &raw_audio, &error);
            check(status, error);
            std::unique_ptr<TtsAudio, decltype(&tts_audio_free)> audio(raw_audio, tts_audio_free);
            const size_t n = tts_audio_length(audio.get());
            const float *pcm = tts_audio_samples(audio.get());
            if (!pcm || n <= 24000 || tts_audio_sample_rate(audio.get()) != 24000)
                throw std::runtime_error("invalid audio shape");
            float peak = 0;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(pcm[i])) throw std::runtime_error("non-finite audio");
                peak = std::max(peak, std::abs(pcm[i]));
            }
            if (peak < 0.001f) throw std::runtime_error("silent audio");
            const auto path = (std::filesystem::path(argv[2]) / (std::string(sample.code) + ".wav")).string();
            error = nullptr;
            status = tts_audio_write_wav(audio.get(), path.c_str(), &error);
            check(status, error);
            std::cout << sample.code << ": " << n << " samples\n";
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
