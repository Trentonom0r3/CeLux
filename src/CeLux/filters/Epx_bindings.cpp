#include "Epx_bindings.hpp"

namespace py = pybind11;

void bind_Epx(py::module_ &m) {
    py::class_<Epx, FilterBase, std::shared_ptr<Epx>>(m, "Epx")
        .def(py::init<int>(),
             py::arg("scaleFactor") = 3)
        .def("setScaleFactor", &Epx::setScaleFactor)
        .def("getScaleFactor", &Epx::getScaleFactor)
        ;
}