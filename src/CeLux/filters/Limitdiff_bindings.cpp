#include "Limitdiff_bindings.hpp"

namespace py = pybind11;

void bind_Limitdiff(py::module_ &m) {
    py::class_<Limitdiff, FilterBase, std::shared_ptr<Limitdiff>>(m, "Limitdiff")
        .def(py::init<float, float, bool, int>(),
             py::arg("threshold") = 0.00,
             py::arg("elasticity") = 2.00,
             py::arg("reference") = false,
             py::arg("planes") = 15)
        .def("setThreshold", &Limitdiff::setThreshold)
        .def("getThreshold", &Limitdiff::getThreshold)
        .def("setElasticity", &Limitdiff::setElasticity)
        .def("getElasticity", &Limitdiff::getElasticity)
        .def("setReference", &Limitdiff::setReference)
        .def("getReference", &Limitdiff::getReference)
        .def("setPlanes", &Limitdiff::setPlanes)
        .def("getPlanes", &Limitdiff::getPlanes)
        ;
}