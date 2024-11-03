#include "Haldclutsrc_bindings.hpp"

namespace py = pybind11;

void bind_Haldclutsrc(py::module_ &m) {
    py::class_<Haldclutsrc, FilterBase, std::shared_ptr<Haldclutsrc>>(m, "Haldclutsrc")
        .def(py::init<int, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("level") = 6,
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setLevel", &Haldclutsrc::setLevel)
        .def("getLevel", &Haldclutsrc::getLevel)
        .def("setRate", &Haldclutsrc::setRate)
        .def("getRate", &Haldclutsrc::getRate)
        .def("setDuration", &Haldclutsrc::setDuration)
        .def("getDuration", &Haldclutsrc::getDuration)
        .def("setSar", &Haldclutsrc::setSar)
        .def("getSar", &Haldclutsrc::getSar)
        ;
}