#include "local_tts.h"
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static void expect(const char *input, uint32_t language, const char *expected) {
    char *raw = nullptr;
    char *error = nullptr;
    const int status = tts_text_prepare(input, language, &raw, &error);
    std::unique_ptr<char, decltype(&tts_string_free)> owned(raw, tts_string_free);
    std::unique_ptr<char, decltype(&tts_error_free)> details(error, tts_error_free);
    assert(status == 0 && raw && !error);
    assert(std::string(raw) == expected);
}
int main() {
    assert(tts_abi_version() == 1);
    expect("2 kg; $5.20", TTS_ENGLISH, "two kilograms; five dollars and twenty cents");
    expect("Дверной замок; 2 кг", TTS_RUSSIAN, "дверной замо́к; два килогра́мма");
    expect("Дверний замок; 2 кг", TTS_UKRAINIAN, "дверний замо́к; два кілогра́ми");
    expect("ÉCOLE ET CŒUR", TTS_FRENCH, "école et cœur");
    expect("GRÜẞE", TTS_GERMAN, "grüße");
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back([] {
        for (int j = 0; j < 20; ++j) expect("21st; 2026-10-03", TTS_ENGLISH,
            "twenty first; october third, two thousand twenty six");
    });
    for (auto &thread : threads) thread.join();
    for (const char *input : {"2026-02-30", "03/10/2026", "v2"}) {
        char *output = nullptr;
        char *error = nullptr;
        assert(tts_text_prepare(input, TTS_ENGLISH, &output, &error) == 1);
        assert(!output && error);
        tts_error_free(error);
    }
    char *output = nullptr;
    char *error = nullptr;
    assert(tts_text_prepare("5", 99, &output, &error) == 1 && !output && error);
    tts_error_free(error);
    tts_string_free(nullptr);
}
