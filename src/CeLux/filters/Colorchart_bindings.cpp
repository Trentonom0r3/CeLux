#include "Colorchart_bindings.hpp"

namespace py = pybind11;

void bind_Colorchart(py::module_ &m) {
    py::class_<Colorchart, FilterBase, std::shared_ptr<Colorchart>>(m, "Colorchart")
        .def(py::init<std::pair<int, int>, int64_t, std::pair<int, int>, std::pair<int, int>, int>(),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1),
             py::arg("patch_size") = std::make_pair<int, int>(0, 1),
             py::arg("preset") = 0)
        .def("setRate", &Colorchart::setRate)
        .def("getRate", &Colorchart::getRate)
        .def("setDuration", &Colorchart::setDuration)
        .def("getDuration", &Colorchart::getDuration)
        .def("setSar", &Colorchart::setSar)
        .def("getSar", &Colorchart::getSar)
        .def("setPatch_size", &Colorchart::setPatch_size)
        .def("getPatch_size", &Colorchart::getPatch_size)
        .def("setPreset", &Colorchart::setPreset)
        .def("getPreset", &Colorchart::getPreset)
        ;
}