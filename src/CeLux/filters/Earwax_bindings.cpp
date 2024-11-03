#include "Earwax_bindings.hpp"

namespace py = pybind11;

void bind_Earwax(py::module_ &m) {
    py::class_<Earwax, FilterBase, std::shared_ptr<Earwax>>(m, "Earwax")
        .def(py::init<>())
        ;
}