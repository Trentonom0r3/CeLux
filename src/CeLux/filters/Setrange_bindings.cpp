#include "Setrange_bindings.hpp"

namespace py = pybind11;

void bind_Setrange(py::module_ &m) {
    py::class_<Setrange, FilterBase, std::shared_ptr<Setrange>>(m, "Setrange")
        .def(py::init<int>(),
             py::arg("range") = -1)
        .def("setRange", &Setrange::setRange)
        .def("getRange", &Setrange::getRange)
        ;
}