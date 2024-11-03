#include "Blurdetect_bindings.hpp"

namespace py = pybind11;

void bind_Blurdetect(py::module_ &m) {
    py::class_<Blurdetect, FilterBase, std::shared_ptr<Blurdetect>>(m, "Blurdetect")
        .def(py::init<float, float, int, int, int, int>(),
             py::arg("high") = 0.12,
             py::arg("low") = 0.06,
             py::arg("radius") = 50,
             py::arg("block_pct") = 80,
             py::arg("block_height") = -1,
             py::arg("planes") = 1)
        .def("setHigh", &Blurdetect::setHigh)
        .def("getHigh", &Blurdetect::getHigh)
        .def("setLow", &Blurdetect::setLow)
        .def("getLow", &Blurdetect::getLow)
        .def("setRadius", &Blurdetect::setRadius)
        .def("getRadius", &Blurdetect::getRadius)
        .def("setBlock_pct", &Blurdetect::setBlock_pct)
        .def("getBlock_pct", &Blurdetect::getBlock_pct)
        .def("setBlock_height", &Blurdetect::setBlock_height)
        .def("getBlock_height", &Blurdetect::getBlock_height)
        .def("setPlanes", &Blurdetect::setPlanes)
        .def("getPlanes", &Blurdetect::getPlanes)
        ;
}