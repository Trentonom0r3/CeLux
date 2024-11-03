#include "Morpho_bindings.hpp"

namespace py = pybind11;

void bind_Morpho(py::module_ &m) {
    py::class_<Morpho, FilterBase, std::shared_ptr<Morpho>>(m, "Morpho")
        .def(py::init<int, int, int>(),
             py::arg("mode") = 0,
             py::arg("planes") = 7,
             py::arg("structure") = 1)
        .def("setMode", &Morpho::setMode)
        .def("getMode", &Morpho::getMode)
        .def("setPlanes", &Morpho::setPlanes)
        .def("getPlanes", &Morpho::getPlanes)
        .def("setStructure", &Morpho::setStructure)
        .def("getStructure", &Morpho::getStructure)
        ;
}