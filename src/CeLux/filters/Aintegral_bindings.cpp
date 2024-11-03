#include "Aintegral_bindings.hpp"

namespace py = pybind11;

void bind_Aintegral(py::module_ &m) {
    py::class_<Aintegral, FilterBase, std::shared_ptr<Aintegral>>(m, "Aintegral")
        .def(py::init<>())
        ;
}