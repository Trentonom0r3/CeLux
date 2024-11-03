#include "Ccrepack_bindings.hpp"

namespace py = pybind11;

void bind_Ccrepack(py::module_ &m) {
    py::class_<Ccrepack, FilterBase, std::shared_ptr<Ccrepack>>(m, "Ccrepack")
        .def(py::init<>())
        ;
}