#include "Idet_bindings.hpp"

namespace py = pybind11;

void bind_Idet(py::module_ &m) {
    py::class_<Idet, FilterBase, std::shared_ptr<Idet>>(m, "Idet")
        .def(py::init<float, float, float, float, int>(),
             py::arg("intl_thres") = 1.04,
             py::arg("prog_thres") = 1.50,
             py::arg("rep_thres") = 3.00,
             py::arg("half_life") = 0.00,
             py::arg("analyze_interlaced_flag") = 0)
        .def("setIntl_thres", &Idet::setIntl_thres)
        .def("getIntl_thres", &Idet::getIntl_thres)
        .def("setProg_thres", &Idet::setProg_thres)
        .def("getProg_thres", &Idet::getProg_thres)
        .def("setRep_thres", &Idet::setRep_thres)
        .def("getRep_thres", &Idet::getRep_thres)
        .def("setHalf_life", &Idet::setHalf_life)
        .def("getHalf_life", &Idet::getHalf_life)
        .def("setAnalyze_interlaced_flag", &Idet::setAnalyze_interlaced_flag)
        .def("getAnalyze_interlaced_flag", &Idet::getAnalyze_interlaced_flag)
        ;
}