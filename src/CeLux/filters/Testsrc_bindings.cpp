#include "Testsrc_bindings.hpp"

namespace py = pybind11;

void bind_Testsrc(py::module_ &m) {
    py::class_<Testsrc, FilterBase, std::shared_ptr<Testsrc>>(m, "Testsrc")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1),
             py::arg("decimals") = 0)
        .def("setSize", &Testsrc::setSize)
        .def("getSize", &Testsrc::getSize)
        .def("setRate", &Testsrc::setRate)
        .def("getRate", &Testsrc::getRate)
        .def("setDuration", &Testsrc::setDuration)
        .def("getDuration", &Testsrc::getDuration)
        .def("setSar", &Testsrc::setSar)
        .def("getSar", &Testsrc::getSar)
        .def("setDecimals", &Testsrc::setDecimals)
        .def("getDecimals", &Testsrc::getDecimals)
        ;
}