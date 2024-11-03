#include "Tiltandshift_bindings.hpp"

namespace py = pybind11;

void bind_Tiltandshift(py::module_ &m) {
    py::class_<Tiltandshift, FilterBase, std::shared_ptr<Tiltandshift>>(m, "Tiltandshift")
        .def(py::init<int, int, int, int, int>(),
             py::arg("tilt") = 1,
             py::arg("start") = 0,
             py::arg("end") = 0,
             py::arg("hold") = 0,
             py::arg("pad") = 0)
        .def("setTilt", &Tiltandshift::setTilt)
        .def("getTilt", &Tiltandshift::getTilt)
        .def("setStart", &Tiltandshift::setStart)
        .def("getStart", &Tiltandshift::getStart)
        .def("setEnd", &Tiltandshift::setEnd)
        .def("getEnd", &Tiltandshift::getEnd)
        .def("setHold", &Tiltandshift::setHold)
        .def("getHold", &Tiltandshift::getHold)
        .def("setPad", &Tiltandshift::setPad)
        .def("getPad", &Tiltandshift::getPad)
        ;
}