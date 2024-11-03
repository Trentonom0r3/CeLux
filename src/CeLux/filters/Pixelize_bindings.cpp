#include "Pixelize_bindings.hpp"

namespace py = pybind11;

void bind_Pixelize(py::module_ &m) {
    py::class_<Pixelize, FilterBase, std::shared_ptr<Pixelize>>(m, "Pixelize")
        .def(py::init<int, int, int, int>(),
             py::arg("width") = 16,
             py::arg("height") = 16,
             py::arg("mode") = 0,
             py::arg("planes") = 15)
        .def("setWidth", &Pixelize::setWidth)
        .def("getWidth", &Pixelize::getWidth)
        .def("setHeight", &Pixelize::setHeight)
        .def("getHeight", &Pixelize::getHeight)
        .def("setMode", &Pixelize::setMode)
        .def("getMode", &Pixelize::getMode)
        .def("setPlanes", &Pixelize::setPlanes)
        .def("getPlanes", &Pixelize::getPlanes)
        ;
}