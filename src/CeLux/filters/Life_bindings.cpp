#include "Life_bindings.hpp"

namespace py = pybind11;

void bind_Life(py::module_ &m) {
    py::class_<Life, FilterBase, std::shared_ptr<Life>>(m, "Life")
        .def(py::init<std::string, std::pair<int, int>, std::pair<int, int>, std::string, double, int64_t, bool, int, std::string, std::string, std::string>(),
             py::arg("filename") = "",
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("rule") = "B3/S23",
             py::arg("random_fill_ratio") = 0.62,
             py::arg("random_seed") = 0,
             py::arg("stitch") = true,
             py::arg("mold") = 0,
             py::arg("life_color") = "white",
             py::arg("death_color") = "black",
             py::arg("mold_color") = "black")
        .def("setFilename", &Life::setFilename)
        .def("getFilename", &Life::getFilename)
        .def("setSize", &Life::setSize)
        .def("getSize", &Life::getSize)
        .def("setRate", &Life::setRate)
        .def("getRate", &Life::getRate)
        .def("setRule", &Life::setRule)
        .def("getRule", &Life::getRule)
        .def("setRandom_fill_ratio", &Life::setRandom_fill_ratio)
        .def("getRandom_fill_ratio", &Life::getRandom_fill_ratio)
        .def("setRandom_seed", &Life::setRandom_seed)
        .def("getRandom_seed", &Life::getRandom_seed)
        .def("setStitch", &Life::setStitch)
        .def("getStitch", &Life::getStitch)
        .def("setMold", &Life::setMold)
        .def("getMold", &Life::getMold)
        .def("setLife_color", &Life::setLife_color)
        .def("getLife_color", &Life::getLife_color)
        .def("setDeath_color", &Life::setDeath_color)
        .def("getDeath_color", &Life::getDeath_color)
        .def("setMold_color", &Life::setMold_color)
        .def("getMold_color", &Life::getMold_color)
        ;
}