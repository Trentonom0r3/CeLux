#include "Framepack_bindings.hpp"

namespace py = pybind11;

void bind_Framepack(py::module_ &m) {
    py::class_<Framepack, FilterBase, std::shared_ptr<Framepack>>(m, "Framepack")
        .def(py::init<int>(),
             py::arg("format") = 1)
        .def("setFormat", &Framepack::setFormat)
        .def("getFormat", &Framepack::getFormat)
        ;
}