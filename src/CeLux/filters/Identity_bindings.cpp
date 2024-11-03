#include "Identity_bindings.hpp"

namespace py = pybind11;

void bind_Identity(py::module_ &m) {
    py::class_<Identity, FilterBase, std::shared_ptr<Identity>>(m, "Identity")
        .def(py::init<>())
        ;
}