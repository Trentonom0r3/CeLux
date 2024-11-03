#include "Corr_bindings.hpp"

namespace py = pybind11;

void bind_Corr(py::module_ &m) {
    py::class_<Corr, FilterBase, std::shared_ptr<Corr>>(m, "Corr")
        .def(py::init<>())
        ;
}