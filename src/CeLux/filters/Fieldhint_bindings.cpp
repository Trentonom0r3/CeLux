#include "Fieldhint_bindings.hpp"

namespace py = pybind11;

void bind_Fieldhint(py::module_ &m) {
    py::class_<Fieldhint, FilterBase, std::shared_ptr<Fieldhint>>(m, "Fieldhint")
        .def(py::init<std::string, int>(),
             py::arg("hint") = "",
             py::arg("mode") = 0)
        .def("setHint", &Fieldhint::setHint)
        .def("getHint", &Fieldhint::getHint)
        .def("setMode", &Fieldhint::setMode)
        .def("getMode", &Fieldhint::getMode)
        ;
}