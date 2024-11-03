#include "Lagfun_bindings.hpp"

namespace py = pybind11;

void bind_Lagfun(py::module_ &m) {
    py::class_<Lagfun, FilterBase, std::shared_ptr<Lagfun>>(m, "Lagfun")
        .def(py::init<float, int>(),
             py::arg("decay") = 0.95,
             py::arg("planes") = 15)
        .def("setDecay", &Lagfun::setDecay)
        .def("getDecay", &Lagfun::getDecay)
        .def("setPlanes", &Lagfun::setPlanes)
        .def("getPlanes", &Lagfun::getPlanes)
        ;
}