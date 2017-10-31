import numpy as np
import pygadgetic

from tqdm import tqdm

from scipy.integrate import quad
from scipy.interpolate import interp1d

#-------------
#You should just be able to import whichever eddington module
#you're interested in and everything should work from here on...
import eddington_exp as edd

#PBH mass and truncation radius imported from the eddington file for
#self-consistency
from eddington_exp import M_PBH, r_tr
#-------------


#Radial distribution function for the DM halo
def P_r_1(r):
        return 4.0*np.pi*r**2*edd.rhoDM(r/r_tr)
        
P_r = np.vectorize(P_r_1)




def AddDressedPBH(body,DMinds, PBHind,nDM, x0, v0):
    """Add a dressed PBH to the initial conditions...
    
    Parameters:
        body - the pygadgetic 'body' object (usually called my_body)
        DMinds - indices specifying where to put the DM particles
                in the list (the DM particles usually come before the PBH
                particles)
        PBHind - single index specifying where to place the PBH
                in the list (close to the end usually, so -1 or -2)
        nDM    - number of DM particles around this PBH
        x0     - initial position of PBH (in units of r_tr)
        v0     - initial velocity of PBH+DM halo (in km/s)
    
    """

    
    #Check that the indices 'inds' are consistent
    #with the number of DM particles
    if (len(DMinds) != nDM):
        print "Error in PBH.AddDressedPBH: number of indices does not match number of particles..."
    
    
    #Generate the mass profile
    print "   Generating mass profile..."
    r_max = 10.0*r_tr
    r_min = 1e-3*r_tr
    
    rlist = np.linspace(r_min, r_max, 100)
    Menc = 0.0*rlist
    for i in range(len(rlist)):
        Menc[i] = quad(P_r, r_min, rlist[i])[0]

    M_max = Menc[-1]
    Minterp = interp1d(Menc/M_max, rlist, kind='linear')
    
    #Calculate and set the pseudo-particle mass
    frac = quad(P_r, r_min, r_tr)[0]/quad(P_r, r_min, 10.0*r_tr)[0]
    m1 = (M_PBH/(frac*nDM))
    body.mass[PBHind] = M_PBH
    body.mass[DMinds] = m1
    
    #PBH position and velocity
    xPBH=np.array([0.,0.,0.])
    vPBH=np.array([0.,0.,0.])
    
    #DM positions
    rvals = Minterp(np.random.rand(nDM))
    
    #Generate some random directions for setting particle positions
    ctvals = 2.0*np.random.rand(nDM) - 1.0
    thetavals = np.arccos(ctvals)
    phivals = 2*np.pi*np.random.rand(nDM)

    xvals = rvals*np.cos(phivals)*np.sin(thetavals)
    yvals = rvals*np.sin(phivals)*np.sin(thetavals)
    zvals = rvals*np.cos(thetavals)

    xDM=np.array([xvals, yvals, zvals]).T

    #DM velocities
    print "   Sampling DM velocities..."
    vvals = np.zeros(nDM)
    for ind in tqdm(range(nDM)):
        r = rvals[ind]
        #Now sample f(v) at given r to get the speed v
        found = 0
        while (found == 0):
            v = np.random.rand(1)*edd.vmax(r/r_tr)
            #Use 5/vmax as the 'maximum' values of f(v)
            #but in some cases it might not be enough...
            if (np.random.rand(1)*(5.0/edd.vmax(r/r_tr)) < edd.f(r, v)):
                found = 1
                vvals[ind] = v

    #Get a new set of random directions for the velocities
    ctvals = 2.0*np.random.rand(nDM) - 1.0
    thetavals = np.arccos(ctvals)
    phivals = 2*np.pi*np.random.rand(nDM)

    vxvals = vvals*np.cos(phivals)*np.sin(thetavals)
    vyvals = vvals*np.sin(phivals)*np.sin(thetavals)
    vzvals = vvals*np.cos(thetavals)

    vDM=np.array([vxvals, vyvals, vzvals]).T

    #Subtract off any net momentum of the system
    totmass = np.sum(body.mass[DMinds])+body.mass[PBHind]
    momentum = np.zeros(3)
    momentum[0] = np.sum(vDM[:,0]*body.mass[DMinds])
    momentum[1] = np.sum(vDM[:,1]*body.mass[DMinds])
    momentum[2] = np.sum(vDM[:,2]*body.mass[DMinds])
    vDM -= momentum/totmass
    vPBH -= momentum/totmass
    
    #Add on the CoM position and velocity
    xDM += np.asarray(x0)*r_tr
    xPBH += np.asarray(x0)*r_tr
    vDM += v0
    vPBH += v0
    
    #Set particle ids
    #body.id[inds]=inds
    
    #Set positions and velocities
    #NB: we divide positions by r_tr
    #to get them in units of...r_tr
    body.pos[PBHind,:] = xPBH/r_tr
    body.vel[PBHind,:] = vPBH
    
    body.pos[DMinds,:] = xDM/r_tr
    body.vel[DMinds,:] = vDM
    
