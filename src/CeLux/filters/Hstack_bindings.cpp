#include "Hstack_bindings.hpp"

namespace py = pybind11;

void bind_Hstack(py::module_ &m) {
    py::class_<Hstack, FilterBase, std::shared_ptr<Hstack>>(m, "Hstack")
        .def(py::init<int, bool>(),
             py::arg("inputs") = 2,
             py::arg("shortest") = false)
        .def("setInputs", &Hstack::setInputs)
        .def("getInputs", &Hstack::getInputs)
        .def("setShortest", &Hstack::setShortest)
        .def("getShortest", &Hstack::getShortest)
        ;
}