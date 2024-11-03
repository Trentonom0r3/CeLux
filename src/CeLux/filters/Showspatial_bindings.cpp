#include "Showspatial_bindings.hpp"

namespace py = pybind11;

void bind_Showspatial(py::module_ &m) {
    py::class_<Showspatial, FilterBase, std::shared_ptr<Showspatial>>(m, "Showspatial")
        .def(py::init<std::pair<int, int>, int, int, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("win_size") = 4096,
             py::arg("win_func") = 1,
             py::arg("rate") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Showspatial::setSize)
        .def("getSize", &Showspatial::getSize)
        .def("setWin_size", &Showspatial::setWin_size)
        .def("getWin_size", &Showspatial::getWin_size)
        .def("setWin_func", &Showspatial::setWin_func)
        .def("getWin_func", &Showspatial::getWin_func)
        .def("setRate", &Showspatial::setRate)
        .def("getRate", &Showspatial::getRate)
        ;
}