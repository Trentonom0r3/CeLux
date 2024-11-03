#include "Anull_bindings.hpp"

namespace py = pybind11;

void bind_Anull(py::module_ &m) {
    py::class_<Anull, FilterBase, std::shared_ptr<Anull>>(m, "Anull")
        .def(py::init<>())
        ;
}