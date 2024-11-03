#include "Vfrdet_bindings.hpp"

namespace py = pybind11;

void bind_Vfrdet(py::module_ &m) {
    py::class_<Vfrdet, FilterBase, std::shared_ptr<Vfrdet>>(m, "Vfrdet")
        .def(py::init<>())
        ;
}