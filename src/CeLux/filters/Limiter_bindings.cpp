#include "Limiter_bindings.hpp"

namespace py = pybind11;

void bind_Limiter(py::module_ &m) {
    py::class_<Limiter, FilterBase, std::shared_ptr<Limiter>>(m, "Limiter")
        .def(py::init<int, int, int>(),
             py::arg("min") = 0,
             py::arg("max") = 65535,
             py::arg("planes") = 15)
        .def("setMin", &Limiter::setMin)
        .def("getMin", &Limiter::getMin)
        .def("setMax", &Limiter::setMax)
        .def("getMax", &Limiter::getMax)
        .def("setPlanes", &Limiter::setPlanes)
        .def("getPlanes", &Limiter::getPlanes)
        ;
}