#include "Premultiply_bindings.hpp"

namespace py = pybind11;

void bind_Premultiply(py::module_ &m) {
    py::class_<Premultiply, FilterBase, std::shared_ptr<Premultiply>>(m, "Premultiply")
        .def(py::init<int, bool>(),
             py::arg("planes") = 15,
             py::arg("inplace") = false)
        .def("setPlanes", &Premultiply::setPlanes)
        .def("getPlanes", &Premultiply::getPlanes)
        .def("setInplace", &Premultiply::setInplace)
        .def("getInplace", &Premultiply::getInplace)
        ;
}