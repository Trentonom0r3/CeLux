#include "Cas_bindings.hpp"

namespace py = pybind11;

void bind_Cas(py::module_ &m) {
    py::class_<Cas, FilterBase, std::shared_ptr<Cas>>(m, "Cas")
        .def(py::init<float, int>(),
             py::arg("strength") = 0.00,
             py::arg("planes") = 7)
        .def("setStrength", &Cas::setStrength)
        .def("getStrength", &Cas::getStrength)
        .def("setPlanes", &Cas::setPlanes)
        .def("getPlanes", &Cas::getPlanes)
        ;
}