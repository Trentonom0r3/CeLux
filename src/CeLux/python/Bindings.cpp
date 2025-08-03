#include "VideoEncoder.hpp"
#include "VideoReader.hpp"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#define PYBIND11_DETAILED_ERROR_MESSAGES

PYBIND11_MODULE(_celux, m)
{
    m.doc() = "celux – lightspeed video decoding into tensors";
     m.attr("__version__") = "0.6.6.9"; 
    m.attr("__all__") = py::make_tuple(
        "__version__",
        "VideoReader",
        "VideoEncoder",
        "Audio", "set_log_level", "LogLevel"
    );
        py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("trace", spdlog::level::trace)
        .value("debug", spdlog::level::debug)
        .value("info", spdlog::level::info)
        .value("warn", spdlog::level::warn)
        .value("error", spdlog::level::err)
        .value("critical", spdlog::level::critical)
        .value("off", spdlog::level::off)
        .export_values();

    m.def("set_log_level", &celux::Logger::set_level, "Set the logging level for CeLux",
          py::arg("level"));
    // ---------- VideoReader -----------
    py::class_<VideoReader, std::shared_ptr<VideoReader>>(m, "VideoReader")
        .def(py::init<const std::string&, int>(), py::arg("input_path"),
             py::arg("num_threads") =
                 static_cast<int>(std::thread::hardware_concurrency() / 2),
             "Open a video file for reading.")
        .def("read_frame", &VideoReader::readFrame,
             "Decode and return the next frame as a H×W×3 uint8 tensor.")
        .def_property_readonly("width", [](VideoReader const& v)
                               { return v.getProperties()["width"].cast<int>(); })
        .def_property_readonly("height", [](VideoReader const& v)
                               { return v.getProperties()["height"].cast<int>(); })
        .def_property_readonly("fps", [](VideoReader const& v)
                               { return v.getProperties()["fps"].cast<double>(); })
        .def_property_readonly("duration", [](VideoReader const& v)
                               { return v.getProperties()["duration"].cast<double>(); })
        .def_property_readonly(
            "total_frames", [](VideoReader const& v)
            { return v.getProperties()["total_frames"].cast<int>(); })
        .def_property_readonly(
            "pixel_format", [](VideoReader const& v)
            { return v.getProperties()["pixel_format"].cast<std::string>(); })
        .def_property_readonly("has_audio", [](VideoReader const& v)
                               { return v.getProperties()["has_audio"].cast<bool>(); })
        .def_property_readonly(
            "audio", &VideoReader::getAudio,
            py::return_value_policy::reference_internal,
            "Get the Audio helper object (with .tensor(), .file(), etc.)")
        .def("__len__", &VideoReader::length)
        .def("__getitem__", &VideoReader::operator[],
             "Index by frame number (int) or timestamp (float).")
        .def(
            "__iter__", [](VideoReader& v) -> VideoReader& { return v.iter(); },
            py::return_value_policy::reference_internal)
        .def("__next__", &VideoReader::next)
        .def("reset", &VideoReader::reset, "Reset reader to beginning or range start.")
        .def("set_range", &VideoReader::setRange, py::arg("start"), py::arg("end"),
             "Set playback range: two ints (frames) or two floats (seconds).")
        .def(
            "create_encoder", &VideoReader::createEncoder, py::arg("output_path"),
            "Create a celux::VideoEncoder configured to this reader's video + audio settings.")
        .def("supported_codecs", &VideoReader::supportedCodecs,
             "List supported decoders.");

    // ----------- Audio Class -----------
    py::class_<VideoReader::Audio, std::shared_ptr<VideoReader::Audio>>(m, "Audio")
        .def("tensor", &VideoReader::Audio::getAudioTensor,
             "Return audio track as a 1-D torch.int16 tensor of interleaved PCM.")
        .def("file", &VideoReader::Audio::extractToFile, py::arg("output_path"),
             "Extract audio to an external file (e.g. WAV)")
        .def_property_readonly("sample_rate", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioSampleRate; })
        .def_property_readonly("channels", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioChannels; })
        .def_property_readonly("bitrate", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioBitrate; })
        .def_property_readonly("codec", [](VideoReader::Audio const& a)
                               { return a.getProperties().audioCodec; });

    // ---------- celux::VideoEncoder -----------
    py::class_<celux::VideoEncoder, std::shared_ptr<celux::VideoEncoder>>(m, "VideoEncoder")
        .def(py::init<const std::string&,         // output_path
                      std::optional<std::string>, // codec
                      std::optional<int>,         // width
                      std::optional<int>,         // height
                      std::optional<int>,         // bit_rate
                      std::optional<float>,       // fps
                      std::optional<int>,         // audio_bit_rate
                      std::optional<int>,         // audio_sample_rate
                      std::optional<int>,         // audio_channels
                      std::optional<std::string>  // audio_codec
                      >(),
             py::arg("output_path"), py::arg("codec") = py::none(),
             py::arg("width") = py::none(), py::arg("height") = py::none(),
             py::arg("bit_rate") = py::none(), py::arg("fps") = py::none(),
             py::arg("audio_bit_rate") = py::none(),
             py::arg("audio_sample_rate") = py::none(),
             py::arg("audio_channels") = py::none(),
             py::arg("audio_codec") = py::none(),
             "Create a celux::VideoEncoder; pass None for defaults.")
        .def("encode_frame", &celux::VideoEncoder::encodeFrame, py::arg("frame"),
             "Encode one video frame (H×W×3 torch.uint8 tensor).")
        .def("encode_audio_frame", &celux::VideoEncoder::encodeAudioFrame, py::arg("audio"),
             "Encode one audio buffer (1-D torch.int16 PCM tensor).")
        .def("close", &celux::VideoEncoder::close,
             "Finalize file and flush audio/video streams.")
        .def(
            "__enter__", [](celux::VideoEncoder& e) -> celux::VideoEncoder& { return e; },
            py::return_value_policy::reference_internal)
        .def("__exit__",
             [](celux::VideoEncoder& e, py::object, py::object, py::object)
             {
                 e.close();
                 return false;
             });
}
