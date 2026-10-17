// Exercise the private native boundary under address/undefined sanitizers.
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" int local_tts_voice_shift(const double *, double *, std::size_t,
                                     unsigned, double, double) noexcept;
static void require(bool value) {
    if (!value) throw std::runtime_error("native shift boundary test failed");
}
int main() {
    try {
        double value = .2, output = 0;
        require(local_tts_voice_shift(nullptr, &output, 1, 24000, 1, 0) == 1);
        require(local_tts_voice_shift(&value, nullptr, 1, 24000, 1, 0) == 1);
        require(local_tts_voice_shift(&value, &output, 0, 24000, 1, 0) == 1);
        require(local_tts_voice_shift(&value, &output, 1, 7999, 1, 0) == 1);
        require(local_tts_voice_shift(&value, &output, 1, 96001, 1, 0) == 1);
        require(local_tts_voice_shift(&value, &output, 1, 24000, NAN, 0) == 1);
        require(local_tts_voice_shift(&value, &output, 1, 24000, 0, INFINITY) == 1);
        for (unsigned rate : {8000u, 24000u, 96000u}) {
            for (int length : {1, 31, 500, 10000}) {
                std::vector<double> input(length + 2, 17);
                std::vector<double> result(length + 2, 19);
                for (int i = 0; i < length; ++i) input[i + 1] = .2 * std::sin(6.283185307179586 * 440 * i / rate);
                const auto original = input;
                for (double pitch : {-12.0, 0.0, 12.0}) {
                    for (double formant : {-6.0, 0.0, 6.0}) {
                        require(local_tts_voice_shift(input.data() + 1, result.data() + 1,
                                                      length, rate, pitch, formant) == 0);
                        require(input == original);
                        require(result.front() == 19 && result.back() == 19);
                        for (int i = 1; i <= length; ++i) require(std::isfinite(result[i]));
                    }
                }
            }
        }
        std::cout << "Native pitch/formant memory bounds and malformed arguments passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
