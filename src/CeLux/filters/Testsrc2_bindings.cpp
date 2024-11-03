#include "Testsrc2_bindings.hpp"

namespace py = pybind11;

void bind_Testsrc2(py::module_ &m) {
    py::class_<Testsrc2, FilterBase, std::shared_ptr<Testsrc2>>(m, "Testsrc2")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1),
             py::arg("alpha") = 255)
        .def("setSize", &Testsrc2::setSize)
        .def("getSize", &Testsrc2::getSize)
        .def("setRate", &Testsrc2::setRate)
        .def("getRate", &Testsrc2::getRate)
        .def("setDuration", &Testsrc2::setDuration)
        .def("getDuration", &Testsrc2::getDuration)
        .def("setSar", &Testsrc2::setSar)
        .def("getSar", &Testsrc2::getSar)
        .def("setAlpha", &Testsrc2::setAlpha)
        .def("getAlpha", &Testsrc2::getAlpha)
        ;
}