#include "Palettegen_bindings.hpp"

namespace py = pybind11;

void bind_Palettegen(py::module_ &m) {
    py::class_<Palettegen, FilterBase, std::shared_ptr<Palettegen>>(m, "Palettegen")
        .def(py::init<int, bool, std::string, int>(),
             py::arg("max_colors") = 256,
             py::arg("reserve_transparent") = true,
             py::arg("transparency_color") = "lime",
             py::arg("stats_mode") = 0)
        .def("setMax_colors", &Palettegen::setMax_colors)
        .def("getMax_colors", &Palettegen::getMax_colors)
        .def("setReserve_transparent", &Palettegen::setReserve_transparent)
        .def("getReserve_transparent", &Palettegen::getReserve_transparent)
        .def("setTransparency_color", &Palettegen::setTransparency_color)
        .def("getTransparency_color", &Palettegen::getTransparency_color)
        .def("setStats_mode", &Palettegen::setStats_mode)
        .def("getStats_mode", &Palettegen::getStats_mode)
        ;
}