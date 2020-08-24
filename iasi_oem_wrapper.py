import numpy as np
import os
from typhon.arts import xml
from typhon.arts.workspace import Workspace
from typhon.physics import wavelength2frequency, wavenumber2frequency, vmr2relative_humidity, e_eq_mixed_mk
import argparse
import sys
sys.path.append(os.getcwd())

import iasi_oem as ioem

# Provide start index of ybatch atmosphere and number of subsequent atmospheres to retrieve in
# first and second command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--ybatch_start", type=int, default=0, help="batch start index")
parser.add_argument("--ybatch_n", type=int, default=1, help="batch profiles per job")
args = parser.parse_args()
ybatch_start = args.ybatch_start
ybatch_n = args.ybatch_n

ws = Workspace(verbosity=1)
project_path = os.getcwd()
ioem.setup_retrieval_paths(project_path)

f_backend_range = np.array([[wavenumber2frequency(1190. * 100),
                             wavenumber2frequency(1400. * 100)],
                            [wavenumber2frequency(2150. * 100),
                             wavenumber2frequency(2400. * 100)]
							])
f_backend_width = wavenumber2frequency(25.)
full_spec = np.arange(wavenumber2frequency(645.0 * 100),
                      wavenumber2frequency(2760.0 * 100),
                      f_backend_width)
garand_path = "../arts/controlfiles/testdata/garand_profiles.xml"
garand = xml.load(garand_path)
z = garand[0][1][:, 0, 0]
p = garand[0].grids[1]

ioem.save_covariances(z=z, p=p,
                      atm_batch_path=garand_path,
                      abs_species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252", "T"])
Sa_T = xml.load(project_path + "/a_priori/covariance_t.xml")
Sa_h2o = xml.load(project_path + "/a_priori/covariance_h2o_vmr.xml")
Sa_cross_h2o_T = ioem.covmat_cross(Sa_h2o, Sa_T, z, corr_height=1000.)
Sa_cross_T_Ts = ioem.covmat_cross(Sa_T, np.array([[100.]]), z, corr_height=100.)
Sa_cross = [
    {'S': Sa_cross_h2o_T,
     'i': 1,
     'j': 2,
     },
    {'S': Sa_cross_T_Ts,
     'i': 0,
     'j': 1,
     },
]

ws = ioem.load_generic_settings(ws)

ws = ioem.setup_sensor(ws, f_backend_width, f_ranges=f_backend_range,
                       add_frequencies=full_spec[np.array([1026, 1190, 1193, 1270, 1883])],)

ws = ioem.load_abs_lookup(ws,
                          abs_lookup_path='../abs_lookup_tables/'
                                          'abs_lookup_garand_profiles_645_2760_cm-1.xml',
                          f_ranges=f_backend_range,
                          )

ws = ioem.iasi_observation(ws,
                           atm_batch_path=garand_path,
                           ybatch_start=ybatch_start,
                           ybatch_n=ybatch_n,
                           add_measurement_noise=True,
                           )

ws = ioem.setup_apriori_state(ws,
                              a_priori_atm_batch_path=garand_path,
                              batch_ind=0,
                              )
ioem.save_z_p_grids(ws, 'a_priori')

ws = ioem.setup_retrieval_quantities(ws,
                                     retrieval_quantities=["H2O", "t_surface", "Temperature"],
                                     cov_t_surface=np.array([[100.]]),
                                     cov_t=Sa_T,
                                     cov_h2o_vmr=Sa_h2o,
                                     cov_cross=Sa_cross,
                                     )
stratospheric_temperature = np.squeeze(garand[0][0]) # T profile for stratosphere
a_priori_rh = vmr2relative_humidity(vmr=garand[0][2][:, 0, 0],
                                    p=garand[0].grids[1],
                                    T=stratospheric_temperature,
                                    e_eq=e_eq_mixed_mk)
ws = ioem.retrieve_ybatch_for_a_priori_batch(ws,
                                             a_priori_atm_batch_path=garand_path,
                                             retrieval_batch_indices=np.arange(ybatch_start, ybatch_start + ybatch_n),
                                             a_priori_batch_indices=np.arange(ybatch_start, ybatch_start + ybatch_n),
                                             retrieval_quantities=["H2O", "t_surface", "Temperature"],
                                             Sa_T=Sa_T,
                                             Sa_t_surface=np.array([[100.]]),
                                             Sa_h2o=Sa_h2o,
                                             Sa_cross=Sa_cross,
                                             moist_adiabat=True,
                                             # t_surface=np.array([[299.0]]),
                                             t_surface_std=1.5,
                                             stratospheric_temperature=stratospheric_temperature,
                                             RH=a_priori_rh,
                                             inversion_method="lm",
                                             max_iter=15,
                                             gamma_start=10,
                                             gamma_dec_factor=2.0,
                                             gamma_inc_factor=2.0,
                                             gamma_upper_limit=1e20,
                                             gamma_lower_limit=1.0,
                                             gamma_upper_convergence_limit=99.0,
                                             )

ioem.plot_profiles(project_path, batch_start=0, batch_end=1)