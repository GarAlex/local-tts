#include "local_tts.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
static void require(bool value) { if (!value) throw std::runtime_error("pronunciation ABI test failed"); }
static void write(const std::string &path, const std::string &bytes) { std::ofstream stream(path); stream << bytes; stream.close(); require(!stream.fail()); }
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    try {
        const bool russian = std::string(argv[2]) == "ru";
        const uint32_t backend = russian ? TTS_VOSK_RU : TTS_PIPER_UK;
        const uint32_t language = russian ? TTS_RUSSIAN : TTS_UKRAINIAN;
        const char *word = russian ? "абрасофатик" : "привіт";
        const char *speaker = russian ? "F03" : "lada";
        const std::string path = std::string(argv[1]) + "/pronunciations.json";
        char *error = nullptr;
        require(tts_engine_reload_pronunciations(nullptr, &error) == 1);
        tts_error_free(error); error = nullptr;
        TtsEngine *raw = nullptr;
        require(tts_engine_load(argv[1], backend, &raw, &error) == 0 && raw && !error);
        std::unique_ptr<TtsEngine, decltype(&tts_engine_free)> engine(raw, tts_engine_free);
        if (russian) {
            TtsAudio *audio = nullptr;
            require(tts_synthesize(engine.get(), word, language, speaker, 1, &audio, &error) == 1 && !audio && error);
            tts_error_free(error); error = nullptr;
        }
        write(path, russian ? R"({"schema_version":1,"language":"ru","entries":{"абрасофатик":"абрасофа́тик"}})" : R"({"schema_version":1,"language":"uk","entries":{"привіт":"приві́т"}})");
        require(tts_engine_reload_pronunciations(engine.get(), &error) == 0 && !error);
        auto synthesize = [&] {
            TtsAudio *raw_audio = nullptr;
            require(tts_synthesize(engine.get(), word, language, speaker, 1, &raw_audio, &error) == 0 && raw_audio && !error);
            std::unique_ptr<TtsAudio, decltype(&tts_audio_free)> audio(raw_audio, tts_audio_free);
            require(tts_audio_sample_rate(audio.get()) == 22050 && tts_audio_length(audio.get()) > 2205);
            const float *pcm = tts_audio_samples(audio.get()); require(pcm);
            for (size_t i = 0; i < tts_audio_length(audio.get()); ++i) require(std::isfinite(pcm[i]));
        };
        synthesize();
        write(path, "malformed external edit");
        require(tts_engine_reload_pronunciations(engine.get(), &error) == 1 && error);
        tts_error_free(error); error = nullptr;
        synthesize(); // Failed reload must retain the prior validated corrections.
        write(path, russian ? R"({"schema_version":1,"language":"uk","entries":{}})" : R"({"schema_version":1,"language":"ru","entries":{}})");
        require(tts_engine_reload_pronunciations(engine.get(), &error) == 1 && error);
        tts_error_free(error); error = nullptr;
        synthesize();
        require(std::remove(path.c_str()) == 0);
        require(tts_engine_reload_pronunciations(engine.get(), &error) == 0 && !error);
        if (russian) {
            TtsAudio *audio = nullptr;
            require(tts_synthesize(engine.get(), word, language, speaker, 1, &audio, &error) == 1 && !audio && error);
            tts_error_free(error);
        } else synthesize();
        std::cout << "C++ " << argv[2] << " correction reload, failed reload retention, language validation and clear passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
