#include "Maskedmerge_bindings.hpp"

namespace py = pybind11;

void bind_Maskedmerge(py::module_ &m) {
    py::class_<Maskedmerge, FilterBase, std::shared_ptr<Maskedmerge>>(m, "Maskedmerge")
        .def(py::init<int>(),
             py::arg("planes") = 15)
        .def("setPlanes", &Maskedmerge::setPlanes)
        .def("getPlanes", &Maskedmerge::getPlanes)
        ;
}