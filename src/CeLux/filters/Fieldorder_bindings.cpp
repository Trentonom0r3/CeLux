#include "Fieldorder_bindings.hpp"

namespace py = pybind11;

void bind_Fieldorder(py::module_ &m) {
    py::class_<Fieldorder, FilterBase, std::shared_ptr<Fieldorder>>(m, "Fieldorder")
        .def(py::init<int>(),
             py::arg("order") = 1)
        .def("setOrder", &Fieldorder::setOrder)
        .def("getOrder", &Fieldorder::getOrder)
        ;
}