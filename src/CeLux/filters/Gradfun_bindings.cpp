#include "Gradfun_bindings.hpp"

namespace py = pybind11;

void bind_Gradfun(py::module_ &m) {
    py::class_<Gradfun, FilterBase, std::shared_ptr<Gradfun>>(m, "Gradfun")
        .def(py::init<float, int>(),
             py::arg("strength") = 1.20,
             py::arg("radius") = 16)
        .def("setStrength", &Gradfun::setStrength)
        .def("getStrength", &Gradfun::getStrength)
        .def("setRadius", &Gradfun::setRadius)
        .def("getRadius", &Gradfun::getRadius)
        ;
}