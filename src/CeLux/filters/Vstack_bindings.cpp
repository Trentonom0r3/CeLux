#include "Vstack_bindings.hpp"

namespace py = pybind11;

void bind_Vstack(py::module_ &m) {
    py::class_<Vstack, FilterBase, std::shared_ptr<Vstack>>(m, "Vstack")
        .def(py::init<int, bool>(),
             py::arg("inputs") = 2,
             py::arg("shortest") = false)
        .def("setInputs", &Vstack::setInputs)
        .def("getInputs", &Vstack::getInputs)
        .def("setShortest", &Vstack::setShortest)
        .def("getShortest", &Vstack::getShortest)
        ;
}