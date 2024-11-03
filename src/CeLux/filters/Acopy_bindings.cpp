#include "Acopy_bindings.hpp"

namespace py = pybind11;

void bind_Acopy(py::module_ &m) {
    py::class_<Acopy, FilterBase, std::shared_ptr<Acopy>>(m, "Acopy")
        .def(py::init<>())
        ;
}