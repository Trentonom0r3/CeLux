#include "Fftdnoiz_bindings.hpp"

namespace py = pybind11;

void bind_Fftdnoiz(py::module_ &m) {
    py::class_<Fftdnoiz, FilterBase, std::shared_ptr<Fftdnoiz>>(m, "Fftdnoiz")
        .def(py::init<float, float, int, float, int, int, int, int, int>(),
             py::arg("sigma") = 1.00,
             py::arg("amount") = 1.00,
             py::arg("block") = 32,
             py::arg("overlap") = 0.50,
             py::arg("method") = 0,
             py::arg("prev") = 0,
             py::arg("next") = 0,
             py::arg("planes") = 7,
             py::arg("window") = 1)
        .def("setSigma", &Fftdnoiz::setSigma)
        .def("getSigma", &Fftdnoiz::getSigma)
        .def("setAmount", &Fftdnoiz::setAmount)
        .def("getAmount", &Fftdnoiz::getAmount)
        .def("setBlock", &Fftdnoiz::setBlock)
        .def("getBlock", &Fftdnoiz::getBlock)
        .def("setOverlap", &Fftdnoiz::setOverlap)
        .def("getOverlap", &Fftdnoiz::getOverlap)
        .def("setMethod", &Fftdnoiz::setMethod)
        .def("getMethod", &Fftdnoiz::getMethod)
        .def("setPrev", &Fftdnoiz::setPrev)
        .def("getPrev", &Fftdnoiz::getPrev)
        .def("setNext", &Fftdnoiz::setNext)
        .def("getNext", &Fftdnoiz::getNext)
        .def("setPlanes", &Fftdnoiz::setPlanes)
        .def("getPlanes", &Fftdnoiz::getPlanes)
        .def("setWindow", &Fftdnoiz::setWindow)
        .def("getWindow", &Fftdnoiz::getWindow)
        ;
}