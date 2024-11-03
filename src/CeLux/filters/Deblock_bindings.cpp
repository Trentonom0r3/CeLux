#include "Deblock_bindings.hpp"

namespace py = pybind11;

void bind_Deblock(py::module_ &m) {
    py::class_<Deblock, FilterBase, std::shared_ptr<Deblock>>(m, "Deblock")
        .def(py::init<int, int, float, float, float, float, int>(),
             py::arg("filter") = 1,
             py::arg("block") = 8,
             py::arg("alpha") = 0.10,
             py::arg("beta") = 0.05,
             py::arg("gamma") = 0.05,
             py::arg("delta") = 0.05,
             py::arg("planes") = 15)
        .def("setFilter", &Deblock::setFilter)
        .def("getFilter", &Deblock::getFilter)
        .def("setBlock", &Deblock::setBlock)
        .def("getBlock", &Deblock::getBlock)
        .def("setAlpha", &Deblock::setAlpha)
        .def("getAlpha", &Deblock::getAlpha)
        .def("setBeta", &Deblock::setBeta)
        .def("getBeta", &Deblock::getBeta)
        .def("setGamma", &Deblock::setGamma)
        .def("getGamma", &Deblock::getGamma)
        .def("setDelta", &Deblock::setDelta)
        .def("getDelta", &Deblock::getDelta)
        .def("setPlanes", &Deblock::setPlanes)
        .def("getPlanes", &Deblock::getPlanes)
        ;
}