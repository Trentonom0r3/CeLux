#include "Showcqt_bindings.hpp"

namespace py = pybind11;

void bind_Showcqt(py::module_ &m) {
    py::class_<Showcqt, FilterBase, std::shared_ptr<Showcqt>>(m, "Showcqt")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int, int, int, bool, std::string, std::string, float, float, float, double, double, double, double, float, std::string, int, int, std::string, std::string, std::string, std::string, bool, int, std::string>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("bar_h") = -1,
             py::arg("axis_h") = -1,
             py::arg("sono_h") = -1,
             py::arg("fullhd") = true,
             py::arg("volume") = "16",
             py::arg("volume2") = "sono_v",
             py::arg("sono_g") = 3.00,
             py::arg("gamma2") = 1.00,
             py::arg("bar_t") = 1.00,
             py::arg("timeclamp") = 0.17,
             py::arg("attack") = 0.00,
             py::arg("basefreq") = 20.02,
             py::arg("endfreq") = 20495.60,
             py::arg("coeffclamp") = 1.00,
             py::arg("tlength") = "384*tc/(384+tc*f)",
             py::arg("count") = 6,
             py::arg("fcount") = 0,
             py::arg("fontfile") = "",
             py::arg("font") = "",
             py::arg("fontcolor") = "st(0, (midi(f)-59.5)/12);st(1, if(between(ld(0),0,1), 0.5-0.5*cos(2*PI*ld(0)), 0));r(1-ld(1)) + b(ld(1))",
             py::arg("axisfile") = "",
             py::arg("text") = true,
             py::arg("csp") = 2,
             py::arg("cscheme") = "1|0.5|0|0|0.5|1")
        .def("setSize", &Showcqt::setSize)
        .def("getSize", &Showcqt::getSize)
        .def("setRate", &Showcqt::setRate)
        .def("getRate", &Showcqt::getRate)
        .def("setBar_h", &Showcqt::setBar_h)
        .def("getBar_h", &Showcqt::getBar_h)
        .def("setAxis_h", &Showcqt::setAxis_h)
        .def("getAxis_h", &Showcqt::getAxis_h)
        .def("setSono_h", &Showcqt::setSono_h)
        .def("getSono_h", &Showcqt::getSono_h)
        .def("setFullhd", &Showcqt::setFullhd)
        .def("getFullhd", &Showcqt::getFullhd)
        .def("setVolume", &Showcqt::setVolume)
        .def("getVolume", &Showcqt::getVolume)
        .def("setVolume2", &Showcqt::setVolume2)
        .def("getVolume2", &Showcqt::getVolume2)
        .def("setSono_g", &Showcqt::setSono_g)
        .def("getSono_g", &Showcqt::getSono_g)
        .def("setGamma2", &Showcqt::setGamma2)
        .def("getGamma2", &Showcqt::getGamma2)
        .def("setBar_t", &Showcqt::setBar_t)
        .def("getBar_t", &Showcqt::getBar_t)
        .def("setTimeclamp", &Showcqt::setTimeclamp)
        .def("getTimeclamp", &Showcqt::getTimeclamp)
        .def("setAttack", &Showcqt::setAttack)
        .def("getAttack", &Showcqt::getAttack)
        .def("setBasefreq", &Showcqt::setBasefreq)
        .def("getBasefreq", &Showcqt::getBasefreq)
        .def("setEndfreq", &Showcqt::setEndfreq)
        .def("getEndfreq", &Showcqt::getEndfreq)
        .def("setCoeffclamp", &Showcqt::setCoeffclamp)
        .def("getCoeffclamp", &Showcqt::getCoeffclamp)
        .def("setTlength", &Showcqt::setTlength)
        .def("getTlength", &Showcqt::getTlength)
        .def("setCount", &Showcqt::setCount)
        .def("getCount", &Showcqt::getCount)
        .def("setFcount", &Showcqt::setFcount)
        .def("getFcount", &Showcqt::getFcount)
        .def("setFontfile", &Showcqt::setFontfile)
        .def("getFontfile", &Showcqt::getFontfile)
        .def("setFont", &Showcqt::setFont)
        .def("getFont", &Showcqt::getFont)
        .def("setFontcolor", &Showcqt::setFontcolor)
        .def("getFontcolor", &Showcqt::getFontcolor)
        .def("setAxisfile", &Showcqt::setAxisfile)
        .def("getAxisfile", &Showcqt::getAxisfile)
        .def("setText", &Showcqt::setText)
        .def("getText", &Showcqt::getText)
        .def("setCsp", &Showcqt::setCsp)
        .def("getCsp", &Showcqt::getCsp)
        .def("setCscheme", &Showcqt::setCscheme)
        .def("getCscheme", &Showcqt::getCscheme)
        ;
}