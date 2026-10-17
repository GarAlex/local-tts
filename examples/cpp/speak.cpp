#include "local_tts.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

static void check(int status, char *error) {
    std::unique_ptr<char, decltype(&tts_error_free)> owned(error, tts_error_free);
    if (status) throw std::runtime_error(error ? error : "TTS failed");
}
int main(int argc, char **argv) {
    if (argc != 3 && argc != 7) {
        std::cerr << "usage: speak <kokoro-folder> <output.wav>\n"
                     "   or: speak <vits-folder> <output.wav> <language:0..9> <speaker> <text> <backend:vits/piper-uk/vosk-ru/chatterbox>\n";
        return 2;
    }
    try {
        if (tts_abi_version() != 1) throw std::runtime_error("incompatible ABI");
        TtsEngine *raw_engine = nullptr;
        char *error = nullptr;
        uint32_t backend = TTS_KOKORO;
        if (argc == 7) {
            const std::string name(argv[6]);
            if (name == "vits") backend = TTS_VITS;
            else if (name == "piper-uk") backend = TTS_PIPER_UK;
            else if (name == "vosk-ru") backend = TTS_VOSK_RU;
            else if (name == "chatterbox") backend = TTS_CHATTERBOX;
            else throw std::runtime_error("unknown backend");
        }
        int status = tts_engine_load(argv[1], backend, &raw_engine, &error);
        check(status, error);
        std::unique_ptr<TtsEngine, decltype(&tts_engine_free)> engine(raw_engine, tts_engine_free);
        TtsAudio *raw_audio = nullptr;
        error = nullptr;
        status = tts_synthesize(engine.get(), argc == 7 ? argv[5] : "Hello world.",
                                argc == 7 ? std::stoul(argv[3]) : TTS_ENGLISH,
                                argc == 7 ? argv[4] : "af_heart", 1.0f, &raw_audio, &error);
        check(status, error);
        std::unique_ptr<TtsAudio, decltype(&tts_audio_free)> audio(raw_audio, tts_audio_free);
        std::cout << tts_audio_length(audio.get()) << " mono samples at "
                  << tts_audio_sample_rate(audio.get()) << " Hz\n";
        error = nullptr;
        status = tts_audio_write_wav(audio.get(), argv[2], &error);
        check(status, error);
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
