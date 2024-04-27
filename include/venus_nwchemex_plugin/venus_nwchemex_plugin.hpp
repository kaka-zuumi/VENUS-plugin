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
DECLARE_PROPERTY_TYPE(MDintegratorPT);
DECLARE_PROPERTY_TYPE(OptimizerPT);
DECLARE_PROPERTY_TYPE(HessianAndFrequencyPT);
DECLARE_PROPERTY_TYPE(BimolecularSamplingPT);
DECLARE_PROPERTY_TYPE(UnimolecularSamplingPT);
DECLARE_PROPERTY_TYPE(ImpactParameterSamplingPT);
DECLARE_PROPERTY_TYPE(AngularMomentumThermalSamplingPT);
DECLARE_PROPERTY_TYPE(VibrationalQuantaThermalSamplingPT);

//-------------------------------Implementations--------------------------------
PROPERTY_TYPE_INPUTS(HessianAndFrequencyPT) {
    using chem_sys_t = const chemist::ChemicalSystem&;
    auto rv = pluginplay::declare_input().add_field<chem_sys_t>("Chemical System");
    rv["Chemical System"].set_description("The chemical system");
    return rv;
}

PROPERTY_TYPE_RESULTS(HessianAndFrequencyPT) {
    using frequencies_t = std::vector<double>;
    using normal_modes_t = std::vector<double>;
    auto rv = pluginplay::declare_result().add_field<frequencies_t>("Frequencies").template add_field<normal_modes_t>("Normal Modes");
    rv["Frequencies"].set_description("The ordered set of frequencies of the system (negative frequencies are imaginary)");
    rv["Normal Modes"].set_description("The set of normal modes ordered by frequency");
    return rv;
}

PROPERTY_TYPE_INPUTS(MDintegratorPT) {
    using chem_sys_t = const chemist::ChemicalSystem&;
    using momenta_t = chemist::PointSet<double>;
    auto rv = pluginplay::declare_input().add_field<chem_sys_t>("Chemical System").template add_field<momenta_t>("Momenta");
    rv["Chemical System"].set_description("The chemical system");
    rv["Momenta"].set_description("The corresponding momenta of this chemical system");
    return rv;
}

PROPERTY_TYPE_RESULTS(MDintegratorPT) {
    using chem_sys_t = chemist::ChemicalSystem;
    using momenta_t = chemist::PointSet<double>;
    auto rv = pluginplay::declare_result().add_field<chem_sys_t>("Time-Evolved Chemical System").template add_field<momenta_t>("Time-Evolved Momenta");
    rv["Time-Evolved Chemical System"].set_description("The chemical system after a single time step");
    rv["Time-Evolved Momenta"].set_description("The corresponding momenta of this chemical system after a single time step");
    return rv;
}

PROPERTY_TYPE_INPUTS(OptimizerPT) {
    using chem_sys_t = const chemist::ChemicalSystem&;
    auto rv = pluginplay::declare_input().add_field<chem_sys_t>("Chemical System");
    rv["Chemical System"].set_description("The chemical system");
    return rv;
}

PROPERTY_TYPE_RESULTS(OptimizerPT) {
    using chem_sys_t = chemist::ChemicalSystem;
    auto rv = pluginplay::declare_result().add_field<chem_sys_t>("Optimized Chemical System");
    rv["Optimized Chemical System"].set_description("The chemical system post-optimization");
    return rv;
}

PROPERTY_TYPE_INPUTS(BimolecularSamplingPT) {
    using chem_sys_tA = const chemist::ChemicalSystem&;
    using chem_sys_tB = const chemist::ChemicalSystem&;
    auto rv = pluginplay::declare_input().add_field<chem_sys_tA>("Chemical System A").template add_field<chem_sys_tB>("Chemical System B");
    rv["Chemical System A"].set_description("The chemical system of molecule A");
    rv["Chemical System B"].set_description("The chemical system of molecule B");
    return rv;
}

PROPERTY_TYPE_RESULTS(BimolecularSamplingPT) {
//  using chem_sys_tA = chemist::ChemicalSystem;
//  using chem_sys_tB = chemist::ChemicalSystem;
//  using momenta_tA = chemist::PointSet<double>;
//  using momenta_tB = chemist::PointSet<double>;
    using chem_sys_t = chemist::ChemicalSystem;
    using momenta_t = chemist::PointSet<double>;
    auto rv = pluginplay::declare_result().add_field<chem_sys_t>("Sampled Chemical System").template add_field<momenta_t>("Sampled Momenta");
    rv["Sampled Chemical System"].set_description("The chemical system from a sampling's distribution");
    rv["Sampled Momenta"].set_description("The corresponding momenta of this chemical system");
//  auto rv = pluginplay::declare_result().add_field<chem_sys_tA>("Sampled Chemical System A").template add_field<momenta_tA>("Sampled Momenta A").template add_field<chem_sys_tB>("Sampled Chemical System B").template add_field<momenta_tB>("Sampled Momenta B");
//  rv["Sampled Chemical System A"].set_description("The chemical system of molecule A from a sampling's distribution");
//  rv["Sampled Chemical System B"].set_description("The chemical system of molecule B from a sampling's distribution");
//  rv["Sampled Momenta A"].set_description("The corresponding momenta of molecule A's chemical system");
//  rv["Sampled Momenta B"].set_description("The corresponding momenta of molecule B's chemical system");
    return rv;
}

PROPERTY_TYPE_INPUTS(UnimolecularSamplingPT) {
    using chem_sys_t = const chemist::ChemicalSystem&;
    auto rv = pluginplay::declare_input().add_field<chem_sys_t>("Chemical System");
    rv["Chemical System"].set_description("The chemical system");
    return rv;
}

PROPERTY_TYPE_RESULTS(UnimolecularSamplingPT) {
    using chem_sys_t = chemist::ChemicalSystem;
    using momenta_t = chemist::PointSet<double>;
    auto rv = pluginplay::declare_result().add_field<chem_sys_t>("Sampled Chemical System").template add_field<momenta_t>("Sampled Momenta");
    rv["Sampled Chemical System"].set_description("The chemical system from a sampling's distribution");
    rv["Sampled Momenta"].set_description("The corresponding momenta of this chemical system");
    return rv;
}

PROPERTY_TYPE_INPUTS(ImpactParameterSamplingPT) {
//  using principal_moments_of_inertia = chemist::Point<double>;
    auto rv = pluginplay::declare_input();

//  rv["Principal Moments of Inertia"].set_description("The diagonal elements (in the order: x, y, z) of the moment of inertia tensor in the principal axes frame");
    return rv;
}

PROPERTY_TYPE_RESULTS(ImpactParameterSamplingPT) {
    using impact_parameter_t = double;
    auto rv = pluginplay::declare_result().add_field<impact_parameter_t>("Sampled Impact Parameter");
    rv["Sampled Impact Parameter"].set_description("The impact parameter sampled from a distribution");
    return rv;
}

PROPERTY_TYPE_INPUTS(AngularMomentumThermalSamplingPT) {
    using principal_moments_of_inertia = chemist::Point<double>;
    auto rv = pluginplay::declare_input().add_field<principal_moments_of_inertia>("Principal Moments of Inertia");

    rv["Principal Moments of Inertia"].set_description("The diagonal elements (in the order: x, y, z) of the moment of inertia tensor in the principal axes frame");
    return rv;
}

PROPERTY_TYPE_RESULTS(AngularMomentumThermalSamplingPT) {
    using rotational_energy_t = double;
    using angular_momenta_t = chemist::Point<double>;
    auto rv = pluginplay::declare_result().add_field<rotational_energy_t>("Sampled Rotational Energy").template add_field<angular_momenta_t>("Sampled Angular Momenta");
    rv["Sampled Angular Momenta"].set_description("The angular momenta from a sampling's thermal distribution");
    rv["Sampled Rotational Energy"].set_description("The corresponding rotational energy of this angular momentum");
    return rv;
}

PROPERTY_TYPE_INPUTS(VibrationalQuantaThermalSamplingPT) {
    using frequency_t = double;
    auto rv = pluginplay::declare_input().add_field<frequency_t>("Vibrational Frequency");

    rv["Vibrational Frequency"].set_description("The vibrational frequency to sample a vibrational quanta from");
    return rv;
}

PROPERTY_TYPE_RESULTS(VibrationalQuantaThermalSamplingPT) {
    using vibrational_quanta_t = int;
    auto rv = pluginplay::declare_result().add_field<vibrational_quanta_t>("Sampled Vibrational Quanta");
    rv["Sampled Vibrational Quanta"].set_description("The vibrational quanta from a sampling's thermal distribution");
    return rv;
}

// Start of the template:

inline void export_sampling(python_module_reference m) {
    EXPORT_PROPERTY_TYPE(MDintegratorPT, m);
    EXPORT_PROPERTY_TYPE(OptimizerPT, m);
    EXPORT_PROPERTY_TYPE(HessianAndFrequencyPT, m);
    EXPORT_PROPERTY_TYPE(BimolecularSamplingPT, m);
    EXPORT_PROPERTY_TYPE(UnimolecularSamplingPT, m);
    EXPORT_PROPERTY_TYPE(ImpactParameterSamplingPT, m);
    EXPORT_PROPERTY_TYPE(AngularMomentumThermalSamplingPT, m);
    EXPORT_PROPERTY_TYPE(VibrationalQuantaThermalSamplingPT, m);
}

// End of the template

} // namespace venus_nwchemex_plugin

