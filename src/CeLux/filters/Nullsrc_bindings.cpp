#include "Nullsrc_bindings.hpp"

namespace py = pybind11;

void bind_Nullsrc(py::module_ &m) {
    py::class_<Nullsrc, FilterBase, std::shared_ptr<Nullsrc>>(m, "Nullsrc")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Nullsrc::setSize)
        .def("getSize", &Nullsrc::getSize)
        .def("setRate", &Nullsrc::setRate)
        .def("getRate", &Nullsrc::getRate)
        .def("setDuration", &Nullsrc::setDuration)
        .def("getDuration", &Nullsrc::getDuration)
        .def("setSar", &Nullsrc::setSar)
        .def("getSar", &Nullsrc::getSar)
        ;
}