#include "Maskedclamp_bindings.hpp"

namespace py = pybind11;

void bind_Maskedclamp(py::module_ &m) {
    py::class_<Maskedclamp, FilterBase, std::shared_ptr<Maskedclamp>>(m, "Maskedclamp")
        .def(py::init<int, int, int>(),
             py::arg("undershoot") = 0,
             py::arg("overshoot") = 0,
             py::arg("planes") = 15)
        .def("setUndershoot", &Maskedclamp::setUndershoot)
        .def("getUndershoot", &Maskedclamp::getUndershoot)
        .def("setOvershoot", &Maskedclamp::setOvershoot)
        .def("getOvershoot", &Maskedclamp::getOvershoot)
        .def("setPlanes", &Maskedclamp::setPlanes)
        .def("getPlanes", &Maskedclamp::getPlanes)
        ;
}