#include "Msad_bindings.hpp"

namespace py = pybind11;

void bind_Msad(py::module_ &m) {
    py::class_<Msad, FilterBase, std::shared_ptr<Msad>>(m, "Msad")
        .def(py::init<>())
        ;
}