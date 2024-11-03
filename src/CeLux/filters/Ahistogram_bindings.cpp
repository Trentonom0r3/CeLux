#include "Ahistogram_bindings.hpp"

namespace py = pybind11;

void bind_Ahistogram(py::module_ &m) {
    py::class_<Ahistogram, FilterBase, std::shared_ptr<Ahistogram>>(m, "Ahistogram")
        .def(py::init<int, std::pair<int, int>, std::pair<int, int>, int, int, int, float, int, int>(),
             py::arg("dmode") = 0,
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("scale") = 3,
             py::arg("ascale") = 1,
             py::arg("acount") = 1,
             py::arg("rheight") = 0.10,
             py::arg("slide") = 0,
             py::arg("hmode") = 0)
        .def("setDmode", &Ahistogram::setDmode)
        .def("getDmode", &Ahistogram::getDmode)
        .def("setRate", &Ahistogram::setRate)
        .def("getRate", &Ahistogram::getRate)
        .def("setSize", &Ahistogram::setSize)
        .def("getSize", &Ahistogram::getSize)
        .def("setScale", &Ahistogram::setScale)
        .def("getScale", &Ahistogram::getScale)
        .def("setAscale", &Ahistogram::setAscale)
        .def("getAscale", &Ahistogram::getAscale)
        .def("setAcount", &Ahistogram::setAcount)
        .def("getAcount", &Ahistogram::getAcount)
        .def("setRheight", &Ahistogram::setRheight)
        .def("getRheight", &Ahistogram::getRheight)
        .def("setSlide", &Ahistogram::setSlide)
        .def("getSlide", &Ahistogram::getSlide)
        .def("setHmode", &Ahistogram::setHmode)
        .def("getHmode", &Ahistogram::getHmode)
        ;
}