#include "Alphaextract_bindings.hpp"

namespace py = pybind11;

void bind_Alphaextract(py::module_ &m) {
    py::class_<Alphaextract, FilterBase, std::shared_ptr<Alphaextract>>(m, "Alphaextract")
        .def(py::init<>())
        ;
}