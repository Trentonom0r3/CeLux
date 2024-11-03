#include "Tile_bindings.hpp"

namespace py = pybind11;

void bind_Tile(py::module_ &m) {
    py::class_<Tile, FilterBase, std::shared_ptr<Tile>>(m, "Tile")
        .def(py::init<std::pair<int, int>, int, int, int, std::string, int, int>(),
             py::arg("layout") = std::make_pair<int, int>(0, 1),
             py::arg("nb_frames") = 0,
             py::arg("margin") = 0,
             py::arg("padding") = 0,
             py::arg("color") = "black",
             py::arg("overlap") = 0,
             py::arg("init_padding") = 0)
        .def("setLayout", &Tile::setLayout)
        .def("getLayout", &Tile::getLayout)
        .def("setNb_frames", &Tile::setNb_frames)
        .def("getNb_frames", &Tile::getNb_frames)
        .def("setMargin", &Tile::setMargin)
        .def("getMargin", &Tile::getMargin)
        .def("setPadding", &Tile::setPadding)
        .def("getPadding", &Tile::getPadding)
        .def("setColor", &Tile::setColor)
        .def("getColor", &Tile::getColor)
        .def("setOverlap", &Tile::setOverlap)
        .def("getOverlap", &Tile::getOverlap)
        .def("setInit_padding", &Tile::setInit_padding)
        .def("getInit_padding", &Tile::getInit_padding)
        ;
}