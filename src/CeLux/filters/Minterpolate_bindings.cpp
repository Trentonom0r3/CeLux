#include "Minterpolate_bindings.hpp"

namespace py = pybind11;

void bind_Minterpolate(py::module_ &m) {
    py::class_<Minterpolate, FilterBase, std::shared_ptr<Minterpolate>>(m, "Minterpolate")
        .def(py::init<std::pair<int, int>, int, int, int, int, int, int, int, int, double>(),
             py::arg("fps") = std::make_pair<int, int>(0, 1),
             py::arg("mi_mode") = 2,
             py::arg("mc_mode") = 0,
             py::arg("me_mode") = 1,
             py::arg("me") = 8,
             py::arg("mb_size") = 16,
             py::arg("search_param") = 32,
             py::arg("vsbmc") = 0,
             py::arg("scd") = 1,
             py::arg("scd_threshold") = 10.00)
        .def("setFps", &Minterpolate::setFps)
        .def("getFps", &Minterpolate::getFps)
        .def("setMi_mode", &Minterpolate::setMi_mode)
        .def("getMi_mode", &Minterpolate::getMi_mode)
        .def("setMc_mode", &Minterpolate::setMc_mode)
        .def("getMc_mode", &Minterpolate::getMc_mode)
        .def("setMe_mode", &Minterpolate::setMe_mode)
        .def("getMe_mode", &Minterpolate::getMe_mode)
        .def("setMe", &Minterpolate::setMe)
        .def("getMe", &Minterpolate::getMe)
        .def("setMb_size", &Minterpolate::setMb_size)
        .def("getMb_size", &Minterpolate::getMb_size)
        .def("setSearch_param", &Minterpolate::setSearch_param)
        .def("getSearch_param", &Minterpolate::getSearch_param)
        .def("setVsbmc", &Minterpolate::setVsbmc)
        .def("getVsbmc", &Minterpolate::getVsbmc)
        .def("setScd", &Minterpolate::setScd)
        .def("getScd", &Minterpolate::getScd)
        .def("setScd_threshold", &Minterpolate::setScd_threshold)
        .def("getScd_threshold", &Minterpolate::getScd_threshold)
        ;
}