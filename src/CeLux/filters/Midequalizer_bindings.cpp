#include "Midequalizer_bindings.hpp"

namespace py = pybind11;

void bind_Midequalizer(py::module_ &m) {
    py::class_<Midequalizer, FilterBase, std::shared_ptr<Midequalizer>>(m, "Midequalizer")
        .def(py::init<int>(),
             py::arg("planes") = 15)
        .def("setPlanes", &Midequalizer::setPlanes)
        .def("getPlanes", &Midequalizer::getPlanes)
        ;
}