#include "Haldclut_bindings.hpp"

namespace py = pybind11;

void bind_Haldclut(py::module_ &m) {
    py::class_<Haldclut, FilterBase, std::shared_ptr<Haldclut>>(m, "Haldclut")
        .def(py::init<int, int>(),
             py::arg("clut") = 1,
             py::arg("interp") = 2)
        .def("setClut", &Haldclut::setClut)
        .def("getClut", &Haldclut::getClut)
        .def("setInterp", &Haldclut::setInterp)
        .def("getInterp", &Haldclut::getInterp)
        ;
}