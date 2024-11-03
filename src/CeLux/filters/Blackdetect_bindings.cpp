#include "Blackdetect_bindings.hpp"

namespace py = pybind11;

void bind_Blackdetect(py::module_ &m) {
    py::class_<Blackdetect, FilterBase, std::shared_ptr<Blackdetect>>(m, "Blackdetect")
        .def(py::init<double, double, double>(),
             py::arg("black_min_duration") = 2.00,
             py::arg("picture_black_ratio_th") = 0.98,
             py::arg("pixel_black_th") = 0.10)
        .def("setBlack_min_duration", &Blackdetect::setBlack_min_duration)
        .def("getBlack_min_duration", &Blackdetect::getBlack_min_duration)
        .def("setPicture_black_ratio_th", &Blackdetect::setPicture_black_ratio_th)
        .def("getPicture_black_ratio_th", &Blackdetect::getPicture_black_ratio_th)
        .def("setPixel_black_th", &Blackdetect::setPixel_black_th)
        .def("getPixel_black_th", &Blackdetect::getPixel_black_th)
        ;
}