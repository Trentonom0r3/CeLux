#include "Separatefields_bindings.hpp"

namespace py = pybind11;

void bind_Separatefields(py::module_ &m) {
    py::class_<Separatefields, FilterBase, std::shared_ptr<Separatefields>>(m, "Separatefields")
        .def(py::init<>())
        ;
}