#include "Alphamerge_bindings.hpp"

namespace py = pybind11;

void bind_Alphamerge(py::module_ &m) {
    py::class_<Alphamerge, FilterBase, std::shared_ptr<Alphamerge>>(m, "Alphamerge")
        .def(py::init<>())
        ;
}