#include "Chromanr_bindings.hpp"

namespace py = pybind11;

void bind_Chromanr(py::module_ &m) {
    py::class_<Chromanr, FilterBase, std::shared_ptr<Chromanr>>(m, "Chromanr")
        .def(py::init<float, int, int, int, int, float, float, float, int>(),
             py::arg("thres") = 30.00,
             py::arg("sizew") = 5,
             py::arg("sizeh") = 5,
             py::arg("stepw") = 1,
             py::arg("steph") = 1,
             py::arg("threy") = 200.00,
             py::arg("threu") = 200.00,
             py::arg("threv") = 200.00,
             py::arg("distance") = 0)
        .def("setThres", &Chromanr::setThres)
        .def("getThres", &Chromanr::getThres)
        .def("setSizew", &Chromanr::setSizew)
        .def("getSizew", &Chromanr::getSizew)
        .def("setSizeh", &Chromanr::setSizeh)
        .def("getSizeh", &Chromanr::getSizeh)
        .def("setStepw", &Chromanr::setStepw)
        .def("getStepw", &Chromanr::getStepw)
        .def("setSteph", &Chromanr::setSteph)
        .def("getSteph", &Chromanr::getSteph)
        .def("setThrey", &Chromanr::setThrey)
        .def("getThrey", &Chromanr::getThrey)
        .def("setThreu", &Chromanr::setThreu)
        .def("getThreu", &Chromanr::getThreu)
        .def("setThrev", &Chromanr::setThrev)
        .def("getThrev", &Chromanr::getThrev)
        .def("setDistance", &Chromanr::setDistance)
        .def("getDistance", &Chromanr::getDistance)
        ;
}