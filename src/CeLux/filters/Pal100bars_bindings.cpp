#include "Pal100bars_bindings.hpp"

namespace py = pybind11;

void bind_Pal100bars(py::module_ &m) {
    py::class_<Pal100bars, FilterBase, std::shared_ptr<Pal100bars>>(m, "Pal100bars")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Pal100bars::setSize)
        .def("getSize", &Pal100bars::getSize)
        .def("setRate", &Pal100bars::setRate)
        .def("getRate", &Pal100bars::getRate)
        .def("setDuration", &Pal100bars::setDuration)
        .def("getDuration", &Pal100bars::getDuration)
        .def("setSar", &Pal100bars::setSar)
        .def("getSar", &Pal100bars::getSar)
        ;
}