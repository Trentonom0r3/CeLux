#include "Apsnr_bindings.hpp"

namespace py = pybind11;

void bind_Apsnr(py::module_ &m) {
    py::class_<Apsnr, FilterBase, std::shared_ptr<Apsnr>>(m, "Apsnr")
        .def(py::init<>())
        ;
}