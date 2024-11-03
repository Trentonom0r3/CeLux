#include "Median_bindings.hpp"

namespace py = pybind11;

void bind_Median(py::module_ &m) {
    py::class_<Median, FilterBase, std::shared_ptr<Median>>(m, "Median")
        .def(py::init<int, int, int, float>(),
             py::arg("radius") = 1,
             py::arg("planes") = 15,
             py::arg("radiusV") = 0,
             py::arg("percentile") = 0.50)
        .def("setRadius", &Median::setRadius)
        .def("getRadius", &Median::getRadius)
        .def("setPlanes", &Median::setPlanes)
        .def("getPlanes", &Median::getPlanes)
        .def("setRadiusV", &Median::setRadiusV)
        .def("getRadiusV", &Median::getRadiusV)
        .def("setPercentile", &Median::setPercentile)
        .def("getPercentile", &Median::getPercentile)
        ;
}