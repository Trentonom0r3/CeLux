#include "Hqx_bindings.hpp"

namespace py = pybind11;

void bind_Hqx(py::module_ &m) {
    py::class_<Hqx, FilterBase, std::shared_ptr<Hqx>>(m, "Hqx")
        .def(py::init<int>(),
             py::arg("scaleFactor") = 3)
        .def("setScaleFactor", &Hqx::setScaleFactor)
        .def("getScaleFactor", &Hqx::getScaleFactor)
        ;
}