#include "Unpremultiply_bindings.hpp"

namespace py = pybind11;

void bind_Unpremultiply(py::module_ &m) {
    py::class_<Unpremultiply, FilterBase, std::shared_ptr<Unpremultiply>>(m, "Unpremultiply")
        .def(py::init<int, bool>(),
             py::arg("planes") = 15,
             py::arg("inplace") = false)
        .def("setPlanes", &Unpremultiply::setPlanes)
        .def("getPlanes", &Unpremultiply::getPlanes)
        .def("setInplace", &Unpremultiply::setInplace)
        .def("getInplace", &Unpremultiply::getInplace)
        ;
}