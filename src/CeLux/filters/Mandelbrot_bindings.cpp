#include "Mandelbrot_bindings.hpp"

namespace py = pybind11;

void bind_Mandelbrot(py::module_ &m) {
    py::class_<Mandelbrot, FilterBase, std::shared_ptr<Mandelbrot>>(m, "Mandelbrot")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int, double, double, double, double, double, double, double, double, double, int, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("maxiter") = 7189,
             py::arg("start_x") = -0.74,
             py::arg("start_y") = -0.13,
             py::arg("start_scale") = 3.00,
             py::arg("end_scale") = 0.30,
             py::arg("end_pts") = 400.00,
             py::arg("bailout") = 10.00,
             py::arg("morphxf") = 0.01,
             py::arg("morphyf") = 0.01,
             py::arg("morphamp") = 0.00,
             py::arg("outer") = 1,
             py::arg("inner") = 3)
        .def("setSize", &Mandelbrot::setSize)
        .def("getSize", &Mandelbrot::getSize)
        .def("setRate", &Mandelbrot::setRate)
        .def("getRate", &Mandelbrot::getRate)
        .def("setMaxiter", &Mandelbrot::setMaxiter)
        .def("getMaxiter", &Mandelbrot::getMaxiter)
        .def("setStart_x", &Mandelbrot::setStart_x)
        .def("getStart_x", &Mandelbrot::getStart_x)
        .def("setStart_y", &Mandelbrot::setStart_y)
        .def("getStart_y", &Mandelbrot::getStart_y)
        .def("setStart_scale", &Mandelbrot::setStart_scale)
        .def("getStart_scale", &Mandelbrot::getStart_scale)
        .def("setEnd_scale", &Mandelbrot::setEnd_scale)
        .def("getEnd_scale", &Mandelbrot::getEnd_scale)
        .def("setEnd_pts", &Mandelbrot::setEnd_pts)
        .def("getEnd_pts", &Mandelbrot::getEnd_pts)
        .def("setBailout", &Mandelbrot::setBailout)
        .def("getBailout", &Mandelbrot::getBailout)
        .def("setMorphxf", &Mandelbrot::setMorphxf)
        .def("getMorphxf", &Mandelbrot::getMorphxf)
        .def("setMorphyf", &Mandelbrot::setMorphyf)
        .def("getMorphyf", &Mandelbrot::getMorphyf)
        .def("setMorphamp", &Mandelbrot::setMorphamp)
        .def("getMorphamp", &Mandelbrot::getMorphamp)
        .def("setOuter", &Mandelbrot::setOuter)
        .def("getOuter", &Mandelbrot::getOuter)
        .def("setInner", &Mandelbrot::setInner)
        .def("getInner", &Mandelbrot::getInner)
        ;
}