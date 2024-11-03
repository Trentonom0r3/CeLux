#include "Pal75bars_bindings.hpp"

namespace py = pybind11;

void bind_Pal75bars(py::module_ &m) {
    py::class_<Pal75bars, FilterBase, std::shared_ptr<Pal75bars>>(m, "Pal75bars")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Pal75bars::setSize)
        .def("getSize", &Pal75bars::getSize)
        .def("setRate", &Pal75bars::setRate)
        .def("getRate", &Pal75bars::getRate)
        .def("setDuration", &Pal75bars::setDuration)
        .def("getDuration", &Pal75bars::getDuration)
        .def("setSar", &Pal75bars::setSar)
        .def("getSar", &Pal75bars::getSar)
        ;
}