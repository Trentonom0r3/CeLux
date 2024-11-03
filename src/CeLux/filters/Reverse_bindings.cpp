#include "Reverse_bindings.hpp"

namespace py = pybind11;

void bind_Reverse(py::module_ &m) {
    py::class_<Reverse, FilterBase, std::shared_ptr<Reverse>>(m, "Reverse")
        .def(py::init<>())
        ;
}