#include "Buffer_bindings.hpp"

namespace py = pybind11;

void bind_Buffer(py::module_ &m) {
    py::class_<Buffer, FilterBase, std::shared_ptr<Buffer>>(m, "Buffer")
        .def(py::init<int, std::pair<int, int>, std::string, std::pair<int, int>, std::pair<int, int>, int, int>(),
             py::arg("height") = 0,
             py::arg("video_size") = std::make_pair<int, int>(0, 1),
             py::arg("pix_fmt") = "",
             py::arg("pixel_aspect") = std::make_pair<int, int>(0, 1),
             py::arg("frame_rate") = std::make_pair<int, int>(0, 1),
             py::arg("colorspace") = 2,
             py::arg("range") = 0)
        .def("setHeight", &Buffer::setHeight)
        .def("getHeight", &Buffer::getHeight)
        .def("setVideo_size", &Buffer::setVideo_size)
        .def("getVideo_size", &Buffer::getVideo_size)
        .def("setPix_fmt", &Buffer::setPix_fmt)
        .def("getPix_fmt", &Buffer::getPix_fmt)
        .def("setPixel_aspect", &Buffer::setPixel_aspect)
        .def("getPixel_aspect", &Buffer::getPixel_aspect)
        .def("setFrame_rate", &Buffer::setFrame_rate)
        .def("getFrame_rate", &Buffer::getFrame_rate)
        .def("setColorspace", &Buffer::setColorspace)
        .def("getColorspace", &Buffer::getColorspace)
        .def("setRange", &Buffer::setRange)
        .def("getRange", &Buffer::getRange)
        ;
}