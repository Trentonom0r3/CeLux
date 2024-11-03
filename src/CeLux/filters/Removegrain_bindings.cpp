#include "Removegrain_bindings.hpp"

namespace py = pybind11;

void bind_Removegrain(py::module_ &m) {
    py::class_<Removegrain, FilterBase, std::shared_ptr<Removegrain>>(m, "Removegrain")
        .def(py::init<int, int, int, int>(),
             py::arg("m0") = 0,
             py::arg("m1") = 0,
             py::arg("m2") = 0,
             py::arg("m3") = 0)
        .def("setM0", &Removegrain::setM0)
        .def("getM0", &Removegrain::getM0)
        .def("setM1", &Removegrain::setM1)
        .def("getM1", &Removegrain::getM1)
        .def("setM2", &Removegrain::setM2)
        .def("getM2", &Removegrain::getM2)
        .def("setM3", &Removegrain::setM3)
        .def("getM3", &Removegrain::getM3)
        ;
}