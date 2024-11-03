#include "Asisdr_bindings.hpp"

namespace py = pybind11;

void bind_Asisdr(py::module_ &m) {
    py::class_<Asisdr, FilterBase, std::shared_ptr<Asisdr>>(m, "Asisdr")
        .def(py::init<>())
        ;
}