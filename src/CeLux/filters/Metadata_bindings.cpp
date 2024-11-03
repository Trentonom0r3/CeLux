#include "Metadata_bindings.hpp"

namespace py = pybind11;

void bind_Metadata(py::module_ &m) {
    py::class_<Metadata, FilterBase, std::shared_ptr<Metadata>>(m, "Metadata")
        .def(py::init<int, std::string, std::string, int, std::string, std::string, bool>(),
             py::arg("mode") = 0,
             py::arg("key") = "",
             py::arg("value") = "",
             py::arg("function") = 0,
             py::arg("expr") = "",
             py::arg("file") = "",
             py::arg("direct") = false)
        .def("setMode", &Metadata::setMode)
        .def("getMode", &Metadata::getMode)
        .def("setKey", &Metadata::setKey)
        .def("getKey", &Metadata::getKey)
        .def("setValue", &Metadata::setValue)
        .def("getValue", &Metadata::getValue)
        .def("setFunction", &Metadata::setFunction)
        .def("getFunction", &Metadata::getFunction)
        .def("setExpr", &Metadata::setExpr)
        .def("getExpr", &Metadata::getExpr)
        .def("setFile", &Metadata::setFile)
        .def("getFile", &Metadata::getFile)
        .def("setDirect", &Metadata::setDirect)
        .def("getDirect", &Metadata::getDirect)
        ;
}