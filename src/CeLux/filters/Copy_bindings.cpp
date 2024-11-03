#include "Copy_bindings.hpp"

namespace py = pybind11;

void bind_Copy(py::module_ &m) {
    py::class_<Copy, FilterBase, std::shared_ptr<Copy>>(m, "Copy")
        .def(py::init<>())
        ;
}