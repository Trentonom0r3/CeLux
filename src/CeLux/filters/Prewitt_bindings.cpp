#include "Prewitt_bindings.hpp"

namespace py = pybind11;

void bind_Prewitt(py::module_ &m) {
    py::class_<Prewitt, FilterBase, std::shared_ptr<Prewitt>>(m, "Prewitt")
        .def(py::init<int, float, float>(),
             py::arg("planes") = 15,
             py::arg("scale") = 1.00,
             py::arg("delta") = 0.00)
        .def("setPlanes", &Prewitt::setPlanes)
        .def("getPlanes", &Prewitt::getPlanes)
        .def("setScale", &Prewitt::setScale)
        .def("getScale", &Prewitt::getScale)
        .def("setDelta", &Prewitt::setDelta)
        .def("getDelta", &Prewitt::getDelta)
        ;
}