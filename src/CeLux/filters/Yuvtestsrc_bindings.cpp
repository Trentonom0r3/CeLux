#include "Yuvtestsrc_bindings.hpp"

namespace py = pybind11;

void bind_Yuvtestsrc(py::module_ &m) {
    py::class_<Yuvtestsrc, FilterBase, std::shared_ptr<Yuvtestsrc>>(m, "Yuvtestsrc")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Yuvtestsrc::setSize)
        .def("getSize", &Yuvtestsrc::getSize)
        .def("setRate", &Yuvtestsrc::setRate)
        .def("getRate", &Yuvtestsrc::getRate)
        .def("setDuration", &Yuvtestsrc::setDuration)
        .def("getDuration", &Yuvtestsrc::getDuration)
        .def("setSar", &Yuvtestsrc::setSar)
        .def("getSar", &Yuvtestsrc::getSar)
        ;
}