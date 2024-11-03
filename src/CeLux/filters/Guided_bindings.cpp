#include "Guided_bindings.hpp"

namespace py = pybind11;

void bind_Guided(py::module_ &m) {
    py::class_<Guided, FilterBase, std::shared_ptr<Guided>>(m, "Guided")
        .def(py::init<int, float, int, int, int, int>(),
             py::arg("radius") = 3,
             py::arg("eps") = 0.01,
             py::arg("mode") = 0,
             py::arg("sub") = 4,
             py::arg("guidance") = 0,
             py::arg("planes") = 1)
        .def("setRadius", &Guided::setRadius)
        .def("getRadius", &Guided::getRadius)
        .def("setEps", &Guided::setEps)
        .def("getEps", &Guided::getEps)
        .def("setMode", &Guided::setMode)
        .def("getMode", &Guided::getMode)
        .def("setSub", &Guided::setSub)
        .def("getSub", &Guided::getSub)
        .def("setGuidance", &Guided::setGuidance)
        .def("getGuidance", &Guided::getGuidance)
        .def("setPlanes", &Guided::setPlanes)
        .def("getPlanes", &Guided::getPlanes)
        ;
}