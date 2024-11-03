#include "Displace_bindings.hpp"

namespace py = pybind11;

void bind_Displace(py::module_ &m) {
    py::class_<Displace, FilterBase, std::shared_ptr<Displace>>(m, "Displace")
        .def(py::init<int>(),
             py::arg("edge") = 1)
        .def("setEdge", &Displace::setEdge)
        .def("getEdge", &Displace::getEdge)
        ;
}