#include "Colorspectrum_bindings.hpp"

namespace py = pybind11;

void bind_Colorspectrum(py::module_ &m) {
    py::class_<Colorspectrum, FilterBase, std::shared_ptr<Colorspectrum>>(m, "Colorspectrum")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1),
             py::arg("type") = 0)
        .def("setSize", &Colorspectrum::setSize)
        .def("getSize", &Colorspectrum::getSize)
        .def("setRate", &Colorspectrum::setRate)
        .def("getRate", &Colorspectrum::getRate)
        .def("setDuration", &Colorspectrum::setDuration)
        .def("getDuration", &Colorspectrum::getDuration)
        .def("setSar", &Colorspectrum::setSar)
        .def("getSar", &Colorspectrum::getSar)
        .def("setType", &Colorspectrum::setType)
        .def("getType", &Colorspectrum::getType)
        ;
}