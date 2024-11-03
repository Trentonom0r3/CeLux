#include "Alatency_bindings.hpp"

namespace py = pybind11;

void bind_Alatency(py::module_ &m) {
    py::class_<Alatency, FilterBase, std::shared_ptr<Alatency>>(m, "Alatency")
        .def(py::init<>())
        ;
}