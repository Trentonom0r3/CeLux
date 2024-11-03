#include "Rgbtestsrc_bindings.hpp"

namespace py = pybind11;

void bind_Rgbtestsrc(py::module_ &m) {
    py::class_<Rgbtestsrc, FilterBase, std::shared_ptr<Rgbtestsrc>>(m, "Rgbtestsrc")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>, bool>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1),
             py::arg("complement") = false)
        .def("setSize", &Rgbtestsrc::setSize)
        .def("getSize", &Rgbtestsrc::getSize)
        .def("setRate", &Rgbtestsrc::setRate)
        .def("getRate", &Rgbtestsrc::getRate)
        .def("setDuration", &Rgbtestsrc::setDuration)
        .def("getDuration", &Rgbtestsrc::getDuration)
        .def("setSar", &Rgbtestsrc::setSar)
        .def("getSar", &Rgbtestsrc::getSar)
        .def("setComplement", &Rgbtestsrc::setComplement)
        .def("getComplement", &Rgbtestsrc::getComplement)
        ;
}