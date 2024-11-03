#include "Framerate_bindings.hpp"

namespace py = pybind11;

void bind_Framerate(py::module_ &m) {
    py::class_<Framerate, FilterBase, std::shared_ptr<Framerate>>(m, "Framerate")
        .def(py::init<std::pair<int, int>, int, int, double, int>(),
             py::arg("fps") = std::make_pair<int, int>(0, 1),
             py::arg("interp_start") = 15,
             py::arg("interp_end") = 240,
             py::arg("scene") = 8.20,
             py::arg("flags") = 1)
        .def("setFps", &Framerate::setFps)
        .def("getFps", &Framerate::getFps)
        .def("setInterp_start", &Framerate::setInterp_start)
        .def("getInterp_start", &Framerate::getInterp_start)
        .def("setInterp_end", &Framerate::setInterp_end)
        .def("getInterp_end", &Framerate::getInterp_end)
        .def("setScene", &Framerate::setScene)
        .def("getScene", &Framerate::getScene)
        .def("setFlags", &Framerate::setFlags)
        .def("getFlags", &Framerate::getFlags)
        ;
}