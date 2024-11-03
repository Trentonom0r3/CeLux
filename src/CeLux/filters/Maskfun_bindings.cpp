#include "Maskfun_bindings.hpp"

namespace py = pybind11;

void bind_Maskfun(py::module_ &m) {
    py::class_<Maskfun, FilterBase, std::shared_ptr<Maskfun>>(m, "Maskfun")
        .def(py::init<int, int, int, int, int>(),
             py::arg("low") = 10,
             py::arg("high") = 10,
             py::arg("planes") = 15,
             py::arg("fill") = 0,
             py::arg("sum") = 10)
        .def("setLow", &Maskfun::setLow)
        .def("getLow", &Maskfun::getLow)
        .def("setHigh", &Maskfun::setHigh)
        .def("getHigh", &Maskfun::getHigh)
        .def("setPlanes", &Maskfun::setPlanes)
        .def("getPlanes", &Maskfun::getPlanes)
        .def("setFill", &Maskfun::setFill)
        .def("getFill", &Maskfun::getFill)
        .def("setSum", &Maskfun::setSum)
        .def("getSum", &Maskfun::getSum)
        ;
}