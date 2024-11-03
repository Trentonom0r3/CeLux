#include "Asdr_bindings.hpp"

namespace py = pybind11;

void bind_Asdr(py::module_ &m) {
    py::class_<Asdr, FilterBase, std::shared_ptr<Asdr>>(m, "Asdr")
        .def(py::init<>())
        ;
}