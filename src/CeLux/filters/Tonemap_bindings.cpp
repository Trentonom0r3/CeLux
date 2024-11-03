#include "Tonemap_bindings.hpp"

namespace py = pybind11;

void bind_Tonemap(py::module_ &m) {
    py::class_<Tonemap, FilterBase, std::shared_ptr<Tonemap>>(m, "Tonemap")
        .def(py::init<int, double, double, double>(),
             py::arg("tonemap") = 0,
             py::arg("param") = 0,
             py::arg("desat") = 2.00,
             py::arg("peak") = 0.00)
        .def("setTonemap", &Tonemap::setTonemap)
        .def("getTonemap", &Tonemap::getTonemap)
        .def("setParam", &Tonemap::setParam)
        .def("getParam", &Tonemap::getParam)
        .def("setDesat", &Tonemap::setDesat)
        .def("getDesat", &Tonemap::getDesat)
        .def("setPeak", &Tonemap::setPeak)
        .def("getPeak", &Tonemap::getPeak)
        ;
}