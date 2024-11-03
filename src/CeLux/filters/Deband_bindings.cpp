#include "Deband_bindings.hpp"

namespace py = pybind11;

void bind_Deband(py::module_ &m) {
    py::class_<Deband, FilterBase, std::shared_ptr<Deband>>(m, "Deband")
        .def(py::init<float, float, float, float, int, float, bool, bool>(),
             py::arg("_1thr") = 0.02,
             py::arg("_2thr") = 0.02,
             py::arg("_3thr") = 0.02,
             py::arg("_4thr") = 0.02,
             py::arg("range") = 16,
             py::arg("direction") = 6.28,
             py::arg("blur") = true,
             py::arg("coupling") = false)
        .def("set_1thr", &Deband::set_1thr)
        .def("get_1thr", &Deband::get_1thr)
        .def("set_2thr", &Deband::set_2thr)
        .def("get_2thr", &Deband::get_2thr)
        .def("set_3thr", &Deband::set_3thr)
        .def("get_3thr", &Deband::get_3thr)
        .def("set_4thr", &Deband::set_4thr)
        .def("get_4thr", &Deband::get_4thr)
        .def("setRange", &Deband::setRange)
        .def("getRange", &Deband::getRange)
        .def("setDirection", &Deband::setDirection)
        .def("getDirection", &Deband::getDirection)
        .def("setBlur", &Deband::setBlur)
        .def("getBlur", &Deband::getBlur)
        .def("setCoupling", &Deband::setCoupling)
        .def("getCoupling", &Deband::getCoupling)
        ;
}