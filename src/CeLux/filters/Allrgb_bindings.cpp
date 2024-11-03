#include "Allrgb_bindings.hpp"

namespace py = pybind11;

void bind_Allrgb(py::module_ &m) {
    py::class_<Allrgb, FilterBase, std::shared_ptr<Allrgb>>(m, "Allrgb")
        .def(py::init<std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setRate", &Allrgb::setRate)
        .def("getRate", &Allrgb::getRate)
        .def("setDuration", &Allrgb::setDuration)
        .def("getDuration", &Allrgb::getDuration)
        .def("setSar", &Allrgb::setSar)
        .def("getSar", &Allrgb::getSar)
        ;
}