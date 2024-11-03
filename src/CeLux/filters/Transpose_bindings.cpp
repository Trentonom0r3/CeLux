#include "Transpose_bindings.hpp"

namespace py = pybind11;

void bind_Transpose(py::module_ &m) {
    py::class_<Transpose, FilterBase, std::shared_ptr<Transpose>>(m, "Transpose")
        .def(py::init<int, int>(),
             py::arg("dir") = 0,
             py::arg("passthrough") = 0)
        .def("setDir", &Transpose::setDir)
        .def("getDir", &Transpose::getDir)
        .def("setPassthrough", &Transpose::setPassthrough)
        .def("getPassthrough", &Transpose::getPassthrough)
        ;
}