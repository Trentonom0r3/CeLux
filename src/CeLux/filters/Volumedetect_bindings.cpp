#include "Volumedetect_bindings.hpp"

namespace py = pybind11;

void bind_Volumedetect(py::module_ &m) {
    py::class_<Volumedetect, FilterBase, std::shared_ptr<Volumedetect>>(m, "Volumedetect")
        .def(py::init<>())
        ;
}