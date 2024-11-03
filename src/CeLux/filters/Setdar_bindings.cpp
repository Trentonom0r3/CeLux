#include "Setdar_bindings.hpp"

namespace py = pybind11;

void bind_Setdar(py::module_ &m) {
    py::class_<Setdar, FilterBase, std::shared_ptr<Setdar>>(m, "Setdar")
        .def(py::init<std::string, int>(),
             py::arg("ratio") = "0",
             py::arg("max") = 100)
        .def("setRatio", &Setdar::setRatio)
        .def("getRatio", &Setdar::getRatio)
        .def("setMax", &Setdar::setMax)
        .def("getMax", &Setdar::getMax)
        ;
}