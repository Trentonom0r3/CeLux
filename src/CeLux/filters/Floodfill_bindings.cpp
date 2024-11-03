#include "Floodfill_bindings.hpp"

namespace py = pybind11;

void bind_Floodfill(py::module_ &m) {
    py::class_<Floodfill, FilterBase, std::shared_ptr<Floodfill>>(m, "Floodfill")
        .def(py::init<int, int, int, int, int, int, int, int, int, int>(),
             py::arg("pixelXCoordinate") = 0,
             py::arg("pixelYCoordinate") = 0,
             py::arg("s0") = 0,
             py::arg("s1") = 0,
             py::arg("s2") = 0,
             py::arg("s3") = 0,
             py::arg("d0") = 0,
             py::arg("d1") = 0,
             py::arg("d2") = 0,
             py::arg("d3") = 0)
        .def("setPixelXCoordinate", &Floodfill::setPixelXCoordinate)
        .def("getPixelXCoordinate", &Floodfill::getPixelXCoordinate)
        .def("setPixelYCoordinate", &Floodfill::setPixelYCoordinate)
        .def("getPixelYCoordinate", &Floodfill::getPixelYCoordinate)
        .def("setS0", &Floodfill::setS0)
        .def("getS0", &Floodfill::getS0)
        .def("setS1", &Floodfill::setS1)
        .def("getS1", &Floodfill::getS1)
        .def("setS2", &Floodfill::setS2)
        .def("getS2", &Floodfill::getS2)
        .def("setS3", &Floodfill::setS3)
        .def("getS3", &Floodfill::getS3)
        .def("setD0", &Floodfill::setD0)
        .def("getD0", &Floodfill::getD0)
        .def("setD1", &Floodfill::setD1)
        .def("getD1", &Floodfill::getD1)
        .def("setD2", &Floodfill::setD2)
        .def("getD2", &Floodfill::getD2)
        .def("setD3", &Floodfill::setD3)
        .def("getD3", &Floodfill::getD3)
        ;
}