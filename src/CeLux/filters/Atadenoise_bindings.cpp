#include "Atadenoise_bindings.hpp"

namespace py = pybind11;

void bind_Atadenoise(py::module_ &m) {
    py::class_<Atadenoise, FilterBase, std::shared_ptr<Atadenoise>>(m, "Atadenoise")
        .def(py::init<float, float, float, float, float, float, int, int, int, float, float, float>(),
             py::arg("_0a") = 0.02,
             py::arg("_0b") = 0.04,
             py::arg("_1a") = 0.02,
             py::arg("_1b") = 0.04,
             py::arg("_2a") = 0.02,
             py::arg("_2b") = 0.04,
             py::arg("howManyFramesToUse") = 9,
             py::arg("whatPlanesToFilter") = 7,
             py::arg("variantOfAlgorithm") = 0,
             py::arg("_0s") = 32767.00,
             py::arg("_1s") = 32767.00,
             py::arg("_2s") = 32767.00)
        .def("set_0a", &Atadenoise::set_0a)
        .def("get_0a", &Atadenoise::get_0a)
        .def("set_0b", &Atadenoise::set_0b)
        .def("get_0b", &Atadenoise::get_0b)
        .def("set_1a", &Atadenoise::set_1a)
        .def("get_1a", &Atadenoise::get_1a)
        .def("set_1b", &Atadenoise::set_1b)
        .def("get_1b", &Atadenoise::get_1b)
        .def("set_2a", &Atadenoise::set_2a)
        .def("get_2a", &Atadenoise::get_2a)
        .def("set_2b", &Atadenoise::set_2b)
        .def("get_2b", &Atadenoise::get_2b)
        .def("setHowManyFramesToUse", &Atadenoise::setHowManyFramesToUse)
        .def("getHowManyFramesToUse", &Atadenoise::getHowManyFramesToUse)
        .def("setWhatPlanesToFilter", &Atadenoise::setWhatPlanesToFilter)
        .def("getWhatPlanesToFilter", &Atadenoise::getWhatPlanesToFilter)
        .def("setVariantOfAlgorithm", &Atadenoise::setVariantOfAlgorithm)
        .def("getVariantOfAlgorithm", &Atadenoise::getVariantOfAlgorithm)
        .def("set_0s", &Atadenoise::set_0s)
        .def("get_0s", &Atadenoise::get_0s)
        .def("set_1s", &Atadenoise::set_1s)
        .def("get_1s", &Atadenoise::get_1s)
        .def("set_2s", &Atadenoise::set_2s)
        .def("get_2s", &Atadenoise::get_2s)
        ;
}