#include "Hwdownload_bindings.hpp"

namespace py = pybind11;

void bind_Hwdownload(py::module_ &m) {
    py::class_<Hwdownload, FilterBase, std::shared_ptr<Hwdownload>>(m, "Hwdownload")
        .def(py::init<>())
        ;
}