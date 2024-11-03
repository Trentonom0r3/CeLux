#include "Kirsch_bindings.hpp"

namespace py = pybind11;

void bind_Kirsch(py::module_ &m) {
    py::class_<Kirsch, FilterBase, std::shared_ptr<Kirsch>>(m, "Kirsch")
        .def(py::init<int, float, float>(),
             py::arg("planes") = 15,
             py::arg("scale") = 1.00,
             py::arg("delta") = 0.00)
        .def("setPlanes", &Kirsch::setPlanes)
        .def("getPlanes", &Kirsch::getPlanes)
        .def("setScale", &Kirsch::setScale)
        .def("getScale", &Kirsch::getScale)
        .def("setDelta", &Kirsch::setDelta)
        .def("getDelta", &Kirsch::getDelta)
        ;
}