#include "Grayworld_bindings.hpp"

namespace py = pybind11;

void bind_Grayworld(py::module_ &m) {
    py::class_<Grayworld, FilterBase, std::shared_ptr<Grayworld>>(m, "Grayworld")
        .def(py::init<>())
        ;
}