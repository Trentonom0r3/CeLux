#include "Gradients_bindings.hpp"

namespace py = pybind11;

void bind_Gradients(py::module_ &m) {
    py::class_<Gradients, FilterBase, std::shared_ptr<Gradients>>(m, "Gradients")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, int, int, int, int, int, int64_t, int64_t, float, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("c0") = "random",
             py::arg("c1") = "random",
             py::arg("c2") = "random",
             py::arg("c3") = "random",
             py::arg("c4") = "random",
             py::arg("c5") = "random",
             py::arg("c6") = "random",
             py::arg("c7") = "random",
             py::arg("x0") = -1,
             py::arg("y0") = -1,
             py::arg("x1") = -1,
             py::arg("y1") = -1,
             py::arg("nb_colors") = 2,
             py::arg("seed") = 0,
             py::arg("duration") = 0,
             py::arg("speed") = 0.01,
             py::arg("type") = 0)
        .def("setSize", &Gradients::setSize)
        .def("getSize", &Gradients::getSize)
        .def("setRate", &Gradients::setRate)
        .def("getRate", &Gradients::getRate)
        .def("setC0", &Gradients::setC0)
        .def("getC0", &Gradients::getC0)
        .def("setC1", &Gradients::setC1)
        .def("getC1", &Gradients::getC1)
        .def("setC2", &Gradients::setC2)
        .def("getC2", &Gradients::getC2)
        .def("setC3", &Gradients::setC3)
        .def("getC3", &Gradients::getC3)
        .def("setC4", &Gradients::setC4)
        .def("getC4", &Gradients::getC4)
        .def("setC5", &Gradients::setC5)
        .def("getC5", &Gradients::getC5)
        .def("setC6", &Gradients::setC6)
        .def("getC6", &Gradients::getC6)
        .def("setC7", &Gradients::setC7)
        .def("getC7", &Gradients::getC7)
        .def("setX0", &Gradients::setX0)
        .def("getX0", &Gradients::getX0)
        .def("setY0", &Gradients::setY0)
        .def("getY0", &Gradients::getY0)
        .def("setX1", &Gradients::setX1)
        .def("getX1", &Gradients::getX1)
        .def("setY1", &Gradients::setY1)
        .def("getY1", &Gradients::getY1)
        .def("setNb_colors", &Gradients::setNb_colors)
        .def("getNb_colors", &Gradients::getNb_colors)
        .def("setSeed", &Gradients::setSeed)
        .def("getSeed", &Gradients::getSeed)
        .def("setDuration", &Gradients::setDuration)
        .def("getDuration", &Gradients::getDuration)
        .def("setSpeed", &Gradients::setSpeed)
        .def("getSpeed", &Gradients::getSpeed)
        .def("setType", &Gradients::setType)
        .def("getType", &Gradients::getType)
        ;
}