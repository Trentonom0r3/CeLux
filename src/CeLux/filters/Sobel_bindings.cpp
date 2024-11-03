#include "Sobel_bindings.hpp"

namespace py = pybind11;

void bind_Sobel(py::module_ &m) {
    py::class_<Sobel, FilterBase, std::shared_ptr<Sobel>>(m, "Sobel")
        .def(py::init<int, float, float>(),
             py::arg("planes") = 15,
             py::arg("scale") = 1.00,
             py::arg("delta") = 0.00)
        .def("setPlanes", &Sobel::setPlanes)
        .def("getPlanes", &Sobel::getPlanes)
        .def("setScale", &Sobel::setScale)
        .def("getScale", &Sobel::getScale)
        .def("setDelta", &Sobel::setDelta)
        .def("getDelta", &Sobel::getDelta)
        ;
}