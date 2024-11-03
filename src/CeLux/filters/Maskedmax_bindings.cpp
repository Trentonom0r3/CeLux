#include "Maskedmax_bindings.hpp"

namespace py = pybind11;

void bind_Maskedmax(py::module_ &m) {
    py::class_<Maskedmax, FilterBase, std::shared_ptr<Maskedmax>>(m, "Maskedmax")
        .def(py::init<int>(),
             py::arg("planes") = 15)
        .def("setPlanes", &Maskedmax::setPlanes)
        .def("getPlanes", &Maskedmax::getPlanes)
        ;
}