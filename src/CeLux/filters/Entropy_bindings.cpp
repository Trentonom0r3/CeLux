#include "Entropy_bindings.hpp"

namespace py = pybind11;

void bind_Entropy(py::module_ &m) {
    py::class_<Entropy, FilterBase, std::shared_ptr<Entropy>>(m, "Entropy")
        .def(py::init<int>(),
             py::arg("mode") = 0)
        .def("setMode", &Entropy::setMode)
        .def("getMode", &Entropy::getMode)
        ;
}