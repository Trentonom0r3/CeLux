#include "Deshake_bindings.hpp"

namespace py = pybind11;

void bind_Deshake(py::module_ &m) {
    py::class_<Deshake, FilterBase, std::shared_ptr<Deshake>>(m, "Deshake")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, std::string, bool>(),
             py::arg("xForTheRectangularSearchArea") = -1,
             py::arg("yForTheRectangularSearchArea") = -1,
             py::arg("widthForTheRectangularSearchArea") = -1,
             py::arg("heightForTheRectangularSearchArea") = -1,
             py::arg("rx") = 16,
             py::arg("ry") = 16,
             py::arg("edge") = 3,
             py::arg("blocksize") = 8,
             py::arg("contrast") = 125,
             py::arg("search") = 0,
             py::arg("filename") = "",
             py::arg("opencl") = false)
        .def("setXForTheRectangularSearchArea", &Deshake::setXForTheRectangularSearchArea)
        .def("getXForTheRectangularSearchArea", &Deshake::getXForTheRectangularSearchArea)
        .def("setYForTheRectangularSearchArea", &Deshake::setYForTheRectangularSearchArea)
        .def("getYForTheRectangularSearchArea", &Deshake::getYForTheRectangularSearchArea)
        .def("setWidthForTheRectangularSearchArea", &Deshake::setWidthForTheRectangularSearchArea)
        .def("getWidthForTheRectangularSearchArea", &Deshake::getWidthForTheRectangularSearchArea)
        .def("setHeightForTheRectangularSearchArea", &Deshake::setHeightForTheRectangularSearchArea)
        .def("getHeightForTheRectangularSearchArea", &Deshake::getHeightForTheRectangularSearchArea)
        .def("setRx", &Deshake::setRx)
        .def("getRx", &Deshake::getRx)
        .def("setRy", &Deshake::setRy)
        .def("getRy", &Deshake::getRy)
        .def("setEdge", &Deshake::setEdge)
        .def("getEdge", &Deshake::getEdge)
        .def("setBlocksize", &Deshake::setBlocksize)
        .def("getBlocksize", &Deshake::getBlocksize)
        .def("setContrast", &Deshake::setContrast)
        .def("getContrast", &Deshake::getContrast)
        .def("setSearch", &Deshake::setSearch)
        .def("getSearch", &Deshake::getSearch)
        .def("setFilename", &Deshake::setFilename)
        .def("getFilename", &Deshake::getFilename)
        .def("setOpencl", &Deshake::setOpencl)
        .def("getOpencl", &Deshake::getOpencl)
        ;
}