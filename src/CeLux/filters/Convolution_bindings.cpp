#include "Convolution_bindings.hpp"

namespace py = pybind11;

void bind_Convolution(py::module_ &m) {
    py::class_<Convolution, FilterBase, std::shared_ptr<Convolution>>(m, "Convolution")
        .def(py::init<std::string, std::string, std::string, std::string, float, float, float, float, float, float, float, float, int, int, int, int>(),
             py::arg("_0m") = "0 0 0 0 1 0 0 0 0",
             py::arg("_1m") = "0 0 0 0 1 0 0 0 0",
             py::arg("_2m") = "0 0 0 0 1 0 0 0 0",
             py::arg("_3m") = "0 0 0 0 1 0 0 0 0",
             py::arg("_0rdiv") = 0.00,
             py::arg("_1rdiv") = 0.00,
             py::arg("_2rdiv") = 0.00,
             py::arg("_3rdiv") = 0.00,
             py::arg("_0bias") = 0.00,
             py::arg("_1bias") = 0.00,
             py::arg("_2bias") = 0.00,
             py::arg("_3bias") = 0.00,
             py::arg("_0mode") = 0,
             py::arg("_1mode") = 0,
             py::arg("_2mode") = 0,
             py::arg("_3mode") = 0)
        .def("set_0m", &Convolution::set_0m)
        .def("get_0m", &Convolution::get_0m)
        .def("set_1m", &Convolution::set_1m)
        .def("get_1m", &Convolution::get_1m)
        .def("set_2m", &Convolution::set_2m)
        .def("get_2m", &Convolution::get_2m)
        .def("set_3m", &Convolution::set_3m)
        .def("get_3m", &Convolution::get_3m)
        .def("set_0rdiv", &Convolution::set_0rdiv)
        .def("get_0rdiv", &Convolution::get_0rdiv)
        .def("set_1rdiv", &Convolution::set_1rdiv)
        .def("get_1rdiv", &Convolution::get_1rdiv)
        .def("set_2rdiv", &Convolution::set_2rdiv)
        .def("get_2rdiv", &Convolution::get_2rdiv)
        .def("set_3rdiv", &Convolution::set_3rdiv)
        .def("get_3rdiv", &Convolution::get_3rdiv)
        .def("set_0bias", &Convolution::set_0bias)
        .def("get_0bias", &Convolution::get_0bias)
        .def("set_1bias", &Convolution::set_1bias)
        .def("get_1bias", &Convolution::get_1bias)
        .def("set_2bias", &Convolution::set_2bias)
        .def("get_2bias", &Convolution::get_2bias)
        .def("set_3bias", &Convolution::set_3bias)
        .def("get_3bias", &Convolution::get_3bias)
        .def("set_0mode", &Convolution::set_0mode)
        .def("get_0mode", &Convolution::get_0mode)
        .def("set_1mode", &Convolution::set_1mode)
        .def("get_1mode", &Convolution::get_1mode)
        .def("set_2mode", &Convolution::set_2mode)
        .def("get_2mode", &Convolution::get_2mode)
        .def("set_3mode", &Convolution::set_3mode)
        .def("get_3mode", &Convolution::get_3mode)
        ;
}