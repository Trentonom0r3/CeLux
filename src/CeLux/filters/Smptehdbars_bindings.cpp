#include "Smptehdbars_bindings.hpp"

namespace py = pybind11;

void bind_Smptehdbars(py::module_ &m) {
    py::class_<Smptehdbars, FilterBase, std::shared_ptr<Smptehdbars>>(m, "Smptehdbars")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Smptehdbars::setSize)
        .def("getSize", &Smptehdbars::getSize)
        .def("setRate", &Smptehdbars::setRate)
        .def("getRate", &Smptehdbars::getRate)
        .def("setDuration", &Smptehdbars::setDuration)
        .def("getDuration", &Smptehdbars::getDuration)
        .def("setSar", &Smptehdbars::setSar)
        .def("getSar", &Smptehdbars::getSar)
        ;
}