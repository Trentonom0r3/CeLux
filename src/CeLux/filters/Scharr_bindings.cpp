#include "Scharr_bindings.hpp"

namespace py = pybind11;

void bind_Scharr(py::module_ &m) {
    py::class_<Scharr, FilterBase, std::shared_ptr<Scharr>>(m, "Scharr")
        .def(py::init<int, float, float>(),
             py::arg("planes") = 15,
             py::arg("scale") = 1.00,
             py::arg("delta") = 0.00)
        .def("setPlanes", &Scharr::setPlanes)
        .def("getPlanes", &Scharr::getPlanes)
        .def("setScale", &Scharr::setScale)
        .def("getScale", &Scharr::getScale)
        .def("setDelta", &Scharr::setDelta)
        .def("getDelta", &Scharr::getDelta)
        ;
}