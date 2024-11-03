#include "Allyuv_bindings.hpp"

namespace py = pybind11;

void bind_Allyuv(py::module_ &m) {
    py::class_<Allyuv, FilterBase, std::shared_ptr<Allyuv>>(m, "Allyuv")
        .def(py::init<std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setRate", &Allyuv::setRate)
        .def("getRate", &Allyuv::getRate)
        .def("setDuration", &Allyuv::setDuration)
        .def("getDuration", &Allyuv::getDuration)
        .def("setSar", &Allyuv::setSar)
        .def("getSar", &Allyuv::getSar)
        ;
}