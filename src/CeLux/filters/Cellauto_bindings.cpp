#include "Cellauto_bindings.hpp"

namespace py = pybind11;

void bind_Cellauto(py::module_ &m) {
    py::class_<Cellauto, FilterBase, std::shared_ptr<Cellauto>>(m, "Cellauto")
        .def(py::init<std::string, std::string, std::pair<int, int>, std::pair<int, int>, int, double, int64_t, bool, bool, bool, bool>(),
             py::arg("filename") = "",
             py::arg("pattern") = "",
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rule") = 110,
             py::arg("random_fill_ratio") = 0.62,
             py::arg("random_seed") = 0,
             py::arg("scroll") = true,
             py::arg("start_full") = false,
             py::arg("full") = true,
             py::arg("stitch") = true)
        .def("setFilename", &Cellauto::setFilename)
        .def("getFilename", &Cellauto::getFilename)
        .def("setPattern", &Cellauto::setPattern)
        .def("getPattern", &Cellauto::getPattern)
        .def("setRate", &Cellauto::setRate)
        .def("getRate", &Cellauto::getRate)
        .def("setSize", &Cellauto::setSize)
        .def("getSize", &Cellauto::getSize)
        .def("setRule", &Cellauto::setRule)
        .def("getRule", &Cellauto::getRule)
        .def("setRandom_fill_ratio", &Cellauto::setRandom_fill_ratio)
        .def("getRandom_fill_ratio", &Cellauto::getRandom_fill_ratio)
        .def("setRandom_seed", &Cellauto::setRandom_seed)
        .def("getRandom_seed", &Cellauto::getRandom_seed)
        .def("setScroll", &Cellauto::setScroll)
        .def("getScroll", &Cellauto::getScroll)
        .def("setStart_full", &Cellauto::setStart_full)
        .def("getStart_full", &Cellauto::getStart_full)
        .def("setFull", &Cellauto::setFull)
        .def("getFull", &Cellauto::getFull)
        .def("setStitch", &Cellauto::setStitch)
        .def("getStitch", &Cellauto::getStitch)
        ;
}