#include "Null_bindings.hpp"

namespace py = pybind11;

void bind_Null(py::module_ &m) {
    py::class_<Null, FilterBase, std::shared_ptr<Null>>(m, "Null")
        .def(py::init<>())
        ;
}