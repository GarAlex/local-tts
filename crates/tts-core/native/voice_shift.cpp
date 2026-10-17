// Original bridge. Vendored Signalsmith headers retain their MIT notices.
#include "signalsmith-stretch.h"
#include <cstddef>
#include <cmath>
#include <limits>
#include <new>

namespace {
struct PaddedInput {
    const double *samples;
    int count, padding;
    struct Channel {
        const PaddedInput *owner;
        double operator[](int index) const {
            index -= owner->padding;
            return index >= 0 && index < owner->count ? owner->samples[index] : 0;
        }
    };
    Channel operator[](int) const { return {this}; }
};
struct CroppedOutput {
    double *samples;
    int count, padding;
    double discarded = 0;
    struct Channel {
        CroppedOutput *owner;
        double &operator[](int index) {
            index -= owner->padding;
            return index >= 0 && index < owner->count ? owner->samples[index] : owner->discarded;
        }
    };
    Channel operator[](int) { return {this}; }
};
}

// Private bridge: Rust validates buffers/settings. No C++ exception or allocator
// crosses the boundary. Padding feeds and drains the spectral windows even for
// very short clips; output is cropped to original sample coordinates.
extern "C" int local_tts_voice_shift(const double *input, double *output,
    std::size_t count, unsigned rate, double pitch, double formant) noexcept {
    try {
        if (!input || !output || !count || rate < 8000 || rate > 96000 ||
            !std::isfinite(pitch) || std::abs(pitch) > 12 ||
            !std::isfinite(formant) || std::abs(formant) > 6) return 1;
        signalsmith::stretch::SignalsmithStretch<double> shift(0L);
        shift.presetDefault(1, double(rate));
        shift.setTransposeSemitones(pitch);
        shift.setFormantSemitones(formant, true); // compensate pitch; shift envelope independently
        shift.setFormantBase(0); // estimate from input rather than assume speaker pitch
        const int padding = shift.seekLength() * 2;
        if (count > std::size_t(std::numeric_limits<int>::max() - 2 * padding)) return 1;
        PaddedInput source{input, int(count), padding};
        CroppedOutput target;
        target.samples = output; target.count = int(count); target.padding = padding;
        return shift.exact(source, int(count) + 2 * padding, target, int(count) + 2 * padding) ? 0 : 1;
    } catch (const std::bad_alloc &) {
        return 2;
    } catch (...) {
        return 1;
    }
}
