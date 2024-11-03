#include "Nullsink_bindings.hpp"

namespace py = pybind11;

void bind_Nullsink(py::module_ &m) {
    py::class_<Nullsink, FilterBase, std::shared_ptr<Nullsink>>(m, "Nullsink")
        .def(py::init<>())
        ;
}