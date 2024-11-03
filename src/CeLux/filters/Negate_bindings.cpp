#include "Negate_bindings.hpp"

namespace py = pybind11;

void bind_Negate(py::module_ &m) {
    py::class_<Negate, FilterBase, std::shared_ptr<Negate>>(m, "Negate")
        .def(py::init<int, bool>(),
             py::arg("components") = 119,
             py::arg("negate_alpha") = false)
        .def("setComponents", &Negate::setComponents)
        .def("getComponents", &Negate::getComponents)
        .def("setNegate_alpha", &Negate::setNegate_alpha)
        .def("getNegate_alpha", &Negate::getNegate_alpha)
        ;
}