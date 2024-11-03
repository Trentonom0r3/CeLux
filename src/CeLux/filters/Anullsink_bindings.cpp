#include "Anullsink_bindings.hpp"

namespace py = pybind11;

void bind_Anullsink(py::module_ &m) {
    py::class_<Anullsink, FilterBase, std::shared_ptr<Anullsink>>(m, "Anullsink")
        .def(py::init<>())
        ;
}