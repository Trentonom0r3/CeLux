#include "Trim_bindings.hpp"

namespace py = pybind11;

void bind_Trim(py::module_ &m) {
    py::class_<Trim, FilterBase, std::shared_ptr<Trim>>(m, "Trim")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(),
             py::arg("starti") = 9223372036854775807ULL,
             py::arg("endi") = 9223372036854775807ULL,
             py::arg("start_pts") = 0,
             py::arg("end_pts") = 0,
             py::arg("durationi") = 0ULL,
             py::arg("start_frame") = 0,
             py::arg("end_frame") = 9223372036854775807ULL)
        .def("setStarti", &Trim::setStarti)
        .def("getStarti", &Trim::getStarti)
        .def("setEndi", &Trim::setEndi)
        .def("getEndi", &Trim::getEndi)
        .def("setStart_pts", &Trim::setStart_pts)
        .def("getStart_pts", &Trim::getStart_pts)
        .def("setEnd_pts", &Trim::setEnd_pts)
        .def("getEnd_pts", &Trim::getEnd_pts)
        .def("setDurationi", &Trim::setDurationi)
        .def("getDurationi", &Trim::getDurationi)
        .def("setStart_frame", &Trim::setStart_frame)
        .def("getStart_frame", &Trim::getStart_frame)
        .def("setEnd_frame", &Trim::setEnd_frame)
        .def("getEnd_frame", &Trim::getEnd_frame)
        ;
}