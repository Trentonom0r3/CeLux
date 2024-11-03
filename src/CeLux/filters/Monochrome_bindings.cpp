#include "Monochrome_bindings.hpp"

namespace py = pybind11;

void bind_Monochrome(py::module_ &m) {
    py::class_<Monochrome, FilterBase, std::shared_ptr<Monochrome>>(m, "Monochrome")
        .def(py::init<float, float, float, float>(),
             py::arg("cb") = 0.00,
             py::arg("cr") = 0.00,
             py::arg("size") = 1.00,
             py::arg("high") = 0.00)
        .def("setCb", &Monochrome::setCb)
        .def("getCb", &Monochrome::getCb)
        .def("setCr", &Monochrome::setCr)
        .def("getCr", &Monochrome::getCr)
        .def("setSize", &Monochrome::setSize)
        .def("getSize", &Monochrome::getSize)
        .def("setHigh", &Monochrome::setHigh)
        .def("getHigh", &Monochrome::getHigh)
        ;
}