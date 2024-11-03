#include "Smptebars_bindings.hpp"

namespace py = pybind11;

void bind_Smptebars(py::module_ &m) {
    py::class_<Smptebars, FilterBase, std::shared_ptr<Smptebars>>(m, "Smptebars")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Smptebars::setSize)
        .def("getSize", &Smptebars::getSize)
        .def("setRate", &Smptebars::setRate)
        .def("getRate", &Smptebars::getRate)
        .def("setDuration", &Smptebars::setDuration)
        .def("getDuration", &Smptebars::getDuration)
        .def("setSar", &Smptebars::setSar)
        .def("getSar", &Smptebars::getSar)
        ;
}