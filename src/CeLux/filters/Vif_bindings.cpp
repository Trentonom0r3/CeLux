#include "Vif_bindings.hpp"

namespace py = pybind11;

void bind_Vif(py::module_ &m) {
    py::class_<Vif, FilterBase, std::shared_ptr<Vif>>(m, "Vif")
        .def(py::init<>())
        ;
}