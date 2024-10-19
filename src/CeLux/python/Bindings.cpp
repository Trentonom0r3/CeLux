#include "Python/VideoReader.hpp"
#include "Python/VideoWriter.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(celux, m)
{
    // VideoReader bindings
    // VideoReader bindings
    #ifndef CUDA_ENABLED
    py::class_<VideoReader>(m, "VideoReader")
        .def(py::init<const std::string&, const std::string&, const std::string&, int>(),
             py::arg("input_path"), py::arg("device") = "cuda",
             py::arg("d_type") = "uint8", py::arg("buffer_size") = 10)
    #else
    py::class_<VideoReader>(m, "VideoReader")
        .def(py::init<const std::string&, const std::string&, const std::string&, int,
                      std::optional<torch::Stream>>(),
             py::arg("input_path"), py::arg("device") = "cuda",
             py::arg("d_type") = "uint8", py::arg("buffer_size") = 10,
             py::arg("stream") = std::nullopt,
             "Initialize a VideoReader with optional CUDA stream")
    #endif
        .def("read_frame", &VideoReader::readFrame)
        .def("seek", &VideoReader::seek)
        .def("supported_codecs", &VideoReader::supportedCodecs)
        .def("get_properties", &VideoReader::getProperties)
        .def("__len__", &VideoReader::length)
        .def(
            "__iter__", [](VideoReader& self) -> VideoReader& { return self.iter(); },
            py::return_value_policy::reference_internal)
        .def("__next__", &VideoReader::next)
        .def(
            "__enter__",
            [](VideoReader& self) -> VideoReader&
            {
                self.enter();
                return self;
            },
            py::return_value_policy::reference_internal)
        .def("__exit__", &VideoReader::exit)
        .def("sync", &VideoReader::sync)
        .def("reset", &VideoReader::reset)
        .def(
            "__call__",
            [](VideoReader& self, py::object arg) -> VideoReader&
            {
                if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg))
                {
                    auto range = arg.cast<std::vector<int>>();
                    if (range.size() != 2)
                    {
                        throw std::runtime_error(
                            "Range must be a list or tuple of two integers");
                    }
                    self.setRange(range[0],
                                  range[1]); // setRange now handles exclusivity and
                                             // negative indices
                }
                else
                {
                    throw std::runtime_error(
                        "Argument must be a list or tuple of two integers");
                }
                return self;
            },
            py::return_value_policy::reference_internal);

        // VideoWriter bindings
        py::class_<VideoWriter>(m, "VideoWriter")
        .def(py::init<const std::string&, int, int, float, const std::string&,
                      const std::string&>(),
             py::arg("file_path"), py::arg("width"), py::arg("height"), py::arg("fps"),
             py::arg("device") = "cuda", py::arg("dtype") = "uint8")
        .def("write_frame", &VideoWriter::writeFrame, py::arg("frame"))
        .def("supported_codecs", &VideoWriter::supportedCodecs)
        .def("__call__", &VideoWriter::writeFrame, py::arg("frame"))
        .def(
            "__enter__", [](VideoWriter& self) -> VideoWriter& { return self; },
            py::return_value_policy::reference_internal)
        .def("__exit__",
             [](VideoWriter& self, py::object exc_type, py::object exc_value,
                py::object traceback)
             {
                 self.close();
                 return false;
             });
}
