#include "Zoompan_bindings.hpp"

namespace py = pybind11;

void bind_Zoompan(py::module_ &m) {
    py::class_<Zoompan, FilterBase, std::shared_ptr<Zoompan>>(m, "Zoompan")
        .def(py::init<std::string, std::string, std::string, std::string, std::pair<int, int>, std::pair<int, int>>(),
             py::arg("zoom") = "1",
             py::arg("x") = "0",
             py::arg("y") = "0",
             py::arg("duration") = "90",
             py::arg("outputImageSize") = std::make_pair<int, int>(0, 1),
             py::arg("fps") = std::make_pair<int, int>(0, 1))
        .def("setZoom", &Zoompan::setZoom)
        .def("getZoom", &Zoompan::getZoom)
        .def("setX", &Zoompan::setX)
        .def("getX", &Zoompan::getX)
        .def("setY", &Zoompan::setY)
        .def("getY", &Zoompan::getY)
        .def("setDuration", &Zoompan::setDuration)
        .def("getDuration", &Zoompan::getDuration)
        .def("setOutputImageSize", &Zoompan::setOutputImageSize)
        .def("getOutputImageSize", &Zoompan::getOutputImageSize)
        .def("setFps", &Zoompan::setFps)
        .def("getFps", &Zoompan::getFps)
        ;
}