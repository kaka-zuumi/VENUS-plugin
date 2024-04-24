#pragma once

//#include <simde/src/python/export_simde.hpp>
//#include <export_simde.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pluginplay/pluginplay.hpp>
//#include <pluginplay/property_type/property_type.hpp>
//#include <pluginplay/property_type/property_type_impl.hpp>
//#include <simde/simde.hpp>
//#include <simde/types.hpp>
#include <chemist/chemist.hpp>

#include "venus_nwchemex_plugin_mm.hpp"

namespace venus_nwchemex_plugin {

// Kazuumi note start
// From: build/_deps/simde-src/src/python/export_simde.hpp

using python_module_type = pybind11::module_;

using python_module_reference = python_module_type&;

template<typename... Args>
using python_class_type = pybind11::class_<Args...>;

// Kazuumi note end


/** @brief The property type for modules that compute the total energy of a
 *         chemical system.
 *
 *  Arguably one of the most important quantities in electronic structure theory
 *  is the total energy of the system. Modules that are capable of computing the
 *  total energy of a chemical system satisfy the TotalEnergy property type.
 *
 */
DECLARE_PROPERTY_TYPE(UnimolecularSamplingPT);

//-------------------------------Implementations--------------------------------
PROPERTY_TYPE_INPUTS(UnimolecularSamplingPT) {
    using chem_sys_t = const chemist::ChemicalSystem&;
    auto rv = pluginplay::declare_input().add_field<chem_sys_t>("Chemical System");
    rv["Chemical System"].set_description("The chemical system");
    return rv;
}

PROPERTY_TYPE_RESULTS(UnimolecularSamplingPT) {
//  auto rv = pluginplay::declare_result().add_field<chemist::ChemicalSystem>("Sampled Chemical System").add_field<chemist::PointSet<double>>("Sampled Momenta");
    using chem_sys_t = chemist::ChemicalSystem;
    using momenta_t = chemist::PointSet<double>;
    auto rv = pluginplay::declare_result().add_field<chem_sys_t>("Sampled Chemical System").template add_field<momenta_t>("Sampled Momenta");
    rv["Sampled Chemical System"].set_description("The chemical system from a sampling's distribution");
    rv["Sampled Momenta"].set_description("The corresponding momenta of this chemical system");
    return rv;
}

// Start of the template:

inline void export_sampling(python_module_reference m) {
    EXPORT_PROPERTY_TYPE(UnimolecularSamplingPT, m);
}

// End of the template

} // namespace venus_nwchemex_plugin

