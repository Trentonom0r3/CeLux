#include "Select_bindings.hpp"

namespace py = pybind11;

void bind_Select(py::module_ &m) {
    py::class_<Select, FilterBase, std::shared_ptr<Select>>(m, "Select")
        .def(py::init<std::string, int>(),
             py::arg("expr") = "1",
             py::arg("outputs") = 1)
        .def("setExpr", &Select::setExpr)
        .def("getExpr", &Select::getExpr)
        .def("setOutputs", &Select::setOutputs)
        .def("getOutputs", &Select::getOutputs)
        ;
}