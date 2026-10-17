fn main() {
    println!("cargo:rerun-if-changed=native");
    cc::Build::new()
        .cpp(true)
        .std("c++11")
        .opt_level(2)
        .file("native/voice_shift.cpp")
        .include("native/vendor/signalsmith-stretch")
        .include("native/vendor/linear/include")
        .compile("local_tts_voice_shift");
}
