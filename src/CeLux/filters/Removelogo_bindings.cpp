#include "Removelogo_bindings.hpp"

namespace py = pybind11;

void bind_Removelogo(py::module_ &m) {
    py::class_<Removelogo, FilterBase, std::shared_ptr<Removelogo>>(m, "Removelogo")
        .def(py::init<std::string>(),
             py::arg("filename") = "")
        .def("setFilename", &Removelogo::setFilename)
        .def("getFilename", &Removelogo::getFilename)
        ;
}