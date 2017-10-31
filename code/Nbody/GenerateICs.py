### this is an example of how to generate an initial condition file

import numpy as np
import pygadgetic
#-------------

import eddington_exp as	edd

#This is the module that add the PBHs to the initial condition file
import PBH

#Number of DM particles per halo
nDM = 1000

##define number of particles
npart=[0,2*nDM,2,0,0,0]
total_number_of_particles=np.sum(npart) #total number of particles


##create objects
my_header=pygadgetic.Header()
my_body=pygadgetic.Body(npart)

a = 15
e = 0.5
apo = (1+e)*a


#PBH+Halo mass
Mhalo = 156.66
G_N2 = edd.G_N/edd.r_tr #Get G_N into correct units (of truncation radii)
mu = G_N2*(2.0*Mhalo)

vapo = np.sqrt(0.5*(1.0-e)*mu/((1.0+e)*a))

PBH.AddDressedPBH(my_body,np.arange(0,nDM),-2, nDM, [-apo/2.0, 0, 0],[0, vapo, 0])
PBH.AddDressedPBH(my_body,np.arange(nDM,2*nDM),-1, nDM, [apo/2.0, 0, 0],[0, -vapo, 0])

##fill in the header
my_header.NumPart_ThisFile = np.array(npart)
my_header.NumPart_Total = np.array(npart)

#id
my_body.id[:]=np.arange(0,total_number_of_particles) #generate an array from 0 to total_number_of_particles

##now writes the initial condition file
my_name="./run/PBH1.dat"
pygadgetic.dump_ic(my_header,my_body,my_name)
