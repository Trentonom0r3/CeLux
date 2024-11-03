#include "Settb_bindings.hpp"

namespace py = pybind11;

void bind_Settb(py::module_ &m) {
    py::class_<Settb, FilterBase, std::shared_ptr<Settb>>(m, "Settb")
        .def(py::init<std::string>(),
             py::arg("expr") = "intb")
        .def("setExpr", &Settb::setExpr)
        .def("getExpr", &Settb::getExpr)
        ;
}