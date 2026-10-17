#include "local_tts.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <string>
#include <algorithm>

static_assert(sizeof(TtsEffects) == 20, "effects layout changed");
static_assert(sizeof(TtsEffectsV2) == 28, "effects v2 layout changed");

static void require(bool value) { if (!value) throw std::runtime_error("ABI test failed"); }

static void test_effects(const TtsAudio *source) {
    const size_t length = tts_audio_length(source);
    const float *pcm = tts_audio_samples(source);
    std::vector<float> original(pcm, pcm + length);
    TtsEffects settings = tts_effects_default();
    require(settings.struct_size == sizeof(TtsEffects));
    require(settings.bass_db == 0 && settings.brightness_db == 0 && settings.compression == 0);
    require(settings.peak_ceiling == .98f);
    TtsAudio *output = nullptr;
    char *error = nullptr;
    require(tts_audio_process(source, &settings, &output, &error) == 0 && output && !error);
    float peak = 0;
    for (float value : original) peak = std::max(peak, std::abs(value));
    const float gain = peak > settings.peak_ceiling ? settings.peak_ceiling / peak : 1;
    require(tts_audio_length(output) == length);
    require(tts_audio_sample_rate(output) == tts_audio_sample_rate(source));
    require(tts_audio_samples(output) != pcm);
    for (size_t i = 0; i < length; ++i) require(std::abs(tts_audio_samples(output)[i] - original[i] * gain) < 1e-6f);
    tts_audio_free(output); output = nullptr;
    for (int invalid = 0; invalid < 8; ++invalid) {
        settings = tts_effects_default();
        if (invalid == 0) settings.struct_size = 0;
        if (invalid == 1) settings.bass_db = NAN;
        if (invalid == 2) settings.brightness_db = 13;
        if (invalid == 3) settings.compression = -1;
        if (invalid == 4) settings.peak_ceiling = 0;
        require(tts_audio_process(invalid == 5 ? nullptr : source,
                                  invalid == 6 ? nullptr : &settings,
                                  invalid == 7 ? nullptr : &output, &error) == 1);
        require(!output && error); tts_error_free(error); error = nullptr;
    }
    // Concurrent effects calls borrow the same immutable input after engine disposal.
    std::atomic<int> processed{0};
    settings = tts_effects_default();
    settings.bass_db = 9; settings.brightness_db = -3; settings.compression = .8f;
    std::vector<std::thread> workers;
    for (int i = 0; i < 2; ++i) workers.emplace_back([&] {
        TtsAudio *result = nullptr;
        char *failure = nullptr;
        if (tts_audio_process(source, &settings, &result, &failure) == 0 && result && !failure) {
            bool valid = tts_audio_length(result) == length;
            bool different = false;
            for (size_t j = 0; valid && j < length; ++j) {
                float value = tts_audio_samples(result)[j];
                valid = std::isfinite(value) && std::abs(value) <= settings.peak_ceiling;
                different = different || std::abs(value - original[j]) > 1e-5f;
            }
            if (valid && different) ++processed;
        }
        tts_audio_free(result); tts_error_free(failure);
    });
    for (auto &worker : workers) worker.join();
    require(processed == 2);
    require(std::equal(original.begin(), original.end(), tts_audio_samples(source)));
    TtsEffectsV2 v2 = tts_effects_v2_default();
    require(v2.struct_size == sizeof(TtsEffectsV2) && v2.pitch_semitones == 0 && v2.formant_semitones == 0);
    require(tts_audio_process_v2(source, &v2, &output, &error) == 0 && output && !error);
    // New neutral API has the same output as the legacy 20-byte settings.
    for (size_t i = 0; i < length; ++i) require(std::abs(tts_audio_samples(output)[i] - original[i] * gain) < 1e-6f);
    tts_audio_free(output); output = nullptr;
    require(tts_audio_process_v2(source, reinterpret_cast<const TtsEffectsV2 *>(&settings), &output, &error) == 1);
    require(!output && error); tts_error_free(error); error = nullptr;
    for (int invalid = 0; invalid < 8; ++invalid) {
        v2 = tts_effects_v2_default();
        if (invalid == 0) v2.pitch_semitones = NAN;
        if (invalid == 1) v2.pitch_semitones = 12.1f;
        if (invalid == 2) v2.formant_semitones = INFINITY;
        if (invalid == 3) v2.formant_semitones = -6.1f;
        if (invalid == 4) v2.struct_size = 20;
        require(tts_audio_process_v2(invalid == 5 ? nullptr : source,
                                     invalid == 6 ? nullptr : &v2,
                                     invalid == 7 ? nullptr : &output, &error) == 1);
        require(!output && error); tts_error_free(error); error = nullptr;
    }
    workers.clear(); processed = 0;
    for (int i = 0; i < 2; ++i) workers.emplace_back([&, i] {
        TtsEffectsV2 config = tts_effects_v2_default();
        config.pitch_semitones = i ? -4 : 4;
        config.formant_semitones = i ? 2 : -2;
        TtsAudio *result = nullptr;
        char *failure = nullptr;
        if (!tts_audio_process_v2(source, &config, &result, &failure) && result && !failure) {
            bool valid = tts_audio_length(result) == length && tts_audio_sample_rate(result) == tts_audio_sample_rate(source);
            bool different = false;
            double energy = 0;
            for (size_t j = 0; valid && j < length; ++j) {
                float value = tts_audio_samples(result)[j];
                valid = std::isfinite(value) && std::abs(value) <= config.peak_ceiling;
                different = different || std::abs(value - original[j]) > 1e-5f;
                energy += double(value) * value;
            }
            if (valid && different && energy > 1e-5) ++processed;
        }
        tts_audio_free(result); tts_error_free(failure);
    });
    for (auto &worker : workers) worker.join();
    require(processed == 2);
    require(std::equal(original.begin(), original.end(), tts_audio_samples(source)));
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        char *error = nullptr;
        TtsEngine *engine = nullptr;
        require(tts_abi_version() == 1);
        require(tts_engine_load(nullptr, TTS_KOKORO, &engine, &error) == 1);
        require(engine == nullptr && error && std::strstr(error, "null"));
        tts_error_free(error);
        error = nullptr;
        require(tts_engine_load(argv[1], 99, &engine, &error) == 1);
        require(engine == nullptr && error);
        tts_error_free(error);
        error = nullptr;
        require(tts_engine_load(argv[1], TTS_KOKORO, &engine, &error) == 0);
        require(engine && !error);
        require(tts_engine_reload_pronunciations(engine, &error) == 1 && error);
        tts_error_free(error); error = nullptr;
        TtsAudio *audio = nullptr;
        TtsCancel *cancel = tts_cancel_new();
        tts_cancel_request(cancel);
        require(tts_synthesize_cancellable(engine, "Hello world.", TTS_ENGLISH, "af_heart", 1, cancel, &audio, &error) == 1);
        require(!audio && error && std::strstr(error, "cancelled"));
        tts_error_free(error); error = nullptr;
        tts_cancel_free(cancel);
        cancel = tts_cancel_new();
        std::string paragraph;
        for (int i = 0; i < 20; ++i) paragraph += "Hello world. This is a local speech test. ";
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            tts_cancel_request(cancel);
        });
        int cancelled = tts_synthesize_cancellable(engine, paragraph.c_str(), TTS_ENGLISH, "af_heart", 1, cancel, &audio, &error);
        canceller.join();
        require(cancelled == 1 && !audio && error && std::strstr(error, "cancelled"));
        tts_error_free(error); error = nullptr;
        tts_cancel_free(cancel);
        require(tts_synthesize(engine, "Hello world.", TTS_RUSSIAN, "af_heart", 1, &audio, &error) == 1);
        require(!audio && error);
        tts_error_free(error);
        error = nullptr;
        require(tts_synthesize(engine, "Hello world.", TTS_ENGLISH, "af_heart", NAN, &audio, &error) == 1);
        require(!audio && error);
        tts_error_free(error);
        // Concurrent calls on one live engine must be serialized safely.
        std::atomic<int> successes{0};
        std::vector<std::thread> workers;
        for (int i = 0; i < 2; ++i) workers.emplace_back([&] {
            TtsAudio *result = nullptr;
            char *failure = nullptr;
            int status = tts_synthesize(engine, "Hello world.", TTS_ENGLISH, "af_heart", 1, &result, &failure);
            if (!status && result && tts_audio_sample_rate(result) == 24000 && tts_audio_length(result) > 2400) {
                const float *pcm = tts_audio_samples(result);
                bool finite = pcm != nullptr;
                for (size_t j = 0; finite && j < tts_audio_length(result); ++j) finite = std::isfinite(pcm[j]);
                if (finite) ++successes;
            }
            tts_audio_free(result);
            tts_error_free(failure);
        });
        for (auto &worker : workers) worker.join();
        require(successes == 2);
        error = nullptr;
        require(tts_synthesize(engine, "Hello world.", TTS_ENGLISH, "af_heart", 1, &audio, &error) == 0 && audio && !error);
        tts_engine_free(engine);
        test_effects(audio);
        tts_audio_free(audio);
        require(tts_audio_samples(nullptr) == nullptr && tts_audio_length(nullptr) == 0 && tts_audio_sample_rate(nullptr) == 0);
        tts_audio_free(nullptr); tts_engine_free(nullptr); tts_error_free(nullptr);
        std::cout << "C++ ABI cancellation, engine reuse, effects, errors, ownership, PCM and concurrency checks passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
