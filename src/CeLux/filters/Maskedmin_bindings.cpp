#include "Maskedmin_bindings.hpp"

namespace py = pybind11;

void bind_Maskedmin(py::module_ &m) {
    py::class_<Maskedmin, FilterBase, std::shared_ptr<Maskedmin>>(m, "Maskedmin")
        .def(py::init<int>(),
             py::arg("planes") = 15)
        .def("setPlanes", &Maskedmin::setPlanes)
        .def("getPlanes", &Maskedmin::getPlanes)
        ;
}