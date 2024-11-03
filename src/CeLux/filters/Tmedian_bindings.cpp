#include "Tmedian_bindings.hpp"

namespace py = pybind11;

void bind_Tmedian(py::module_ &m) {
    py::class_<Tmedian, FilterBase, std::shared_ptr<Tmedian>>(m, "Tmedian")
        .def(py::init<int, int, float>(),
             py::arg("radius") = 1,
             py::arg("planes") = 15,
             py::arg("percentile") = 0.50)
        .def("setRadius", &Tmedian::setRadius)
        .def("getRadius", &Tmedian::getRadius)
        .def("setPlanes", &Tmedian::setPlanes)
        .def("getPlanes", &Tmedian::getPlanes)
        .def("setPercentile", &Tmedian::setPercentile)
        .def("getPercentile", &Tmedian::getPercentile)
        ;
}