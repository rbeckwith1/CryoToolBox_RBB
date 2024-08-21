"""Pressure drop and heat transfer calculation.
"""

from math import pi, sin, log, log10, sqrt, tan, tanh     
from .std_conditions import ureg, Q_, P_NTP
from .functions import Re, Ra, Nu_vcyl, Nu_hcyl     
from .functions import Material, Property     
from .functions import nist_property, conduction_cyl     
from scipy.optimize import root_scalar, minimize  
from .functions import AIR
from .functions import heat_trans_coef, Ra, Nu_vcyl, Nu_hcyl, Pr
from .cp_wrapper import ThermState
from .piping import Mach, Mach_total, K_lim, ChokedFlow, HydraulicError, velocity, dP_Darcy, dP_adiab, Pipe, Tube, CopperTube
import numpy as np

class pipe_isolation:
    ###  class to define the necessary isolation imputs
    def __init__(self, k, OD, T_ext = 293 * ureg.K):
        self.k = k
        self.OD = OD
        self.T_ext = T_ext

def laminar_flow(Re_, Pr_, L_ID):
    # Non dimentional calcul of the Nusselt and fiction factor in pipe in laminar flow following Section 5.2.4 of Nellis and Klein (2020)
    # Verify the input conditions     
    if Pr_ < 0.1:
        raise ValueError(f'Prandtl number (Pr) must be > 0.1. The value is {Pr}')
        
    #Inverse Graetz numbers verification
    GZ = L_ID / (Re_ * Pr_)
    SGZ = L_ID / Re_
    if GZ < 1e-6:
        raise ValueError(f'Inverse Graetz number (GZ) must be > 1e-6. The value is {GZ}')
         
    # Calculate friction factor (f)
    f = 4 * (3.44 / sqrt(SGZ) + (1.25 / (4 * SGZ) + 16 - 3.44 / sqrt(SGZ)) / (1 + 0.00021 * SGZ**(-2))) / Re_
    
    # Calculate Nusselt numbers, temperature constant and flux constant
    Nu_T = ((5.001 / GZ**1.119 + 136.0)**0.2978 - 0.6628) / tanh(2.444 * SGZ**(1 / 6) * (1 + 0.565 * SGZ**(1 / 3)))
    Nu_Q = ((6.562 / GZ**1.137 + 220.4)**0.2932 - 0.5003) / tanh(2.530 * SGZ**(1 / 6) * (1 + 0.639 * SGZ**(1 / 3)))
    
    return Nu_T, Nu_Q, f

def turbulent_flow(Re_, Pr_, L_ID, eps):
    # Non dimentional calcul of the Nusselt and fiction factor in pipe in turbulent flow following Section 5.2.3 of Nellis and Klein (2020)
    # Verify the input conditions
    if Pr_ < 0.004 or Pr_ > 2000:
        raise ValueError(f'Prandtl number (Pr) must be between 0.004 and 2000. The value is {Pr}')
    if L_ID <= 1:  
        if L_ID < 0:  ###not inferior to zero - make no sense
            raise ValueError('L/ID ratio < 0. Not possible')
        print('L/ID ratio should be > 1. The value is {L_ID}')
        L_ID = 1
        
    # Friction Factor 
    if eps > 1e-5:
        #Offor & Alabi, Advances in Chemical Engineering and Science, 2016
        friction = (-2 * log10(eps / 3.71 - 1.975 / Re_ * log((eps / 3.93)**1.092 + 7.627 / (Re_ + 395.9))))**(-2)
    else:
        #Li & Seem correlation, A New Explicity Equation for Accurate Friction Factor Calculation for Smooth Tubes, 2011
        friction = (-0.001570232 / log(Re_) + 0.394203137 / log(Re_)**2 + 2.534153311 / log(Re_)**3) * 4
        
    # Nusselt, Gnielinski, Int. Chem. Eng., 1976
    Nusselt = ((friction / 8) * (Re_ - 1000) * Pr_) / (1 + 12.7 * sqrt(friction / 8) * (Pr_**(2 / 3) - 1))
    
    # Correct Nusselt number for low Prandtl numbers, Notter & Sleicher, Chem. Eng. Sci., 1972
    if Pr_ < 0.5:
        Nusselt_lp = 4.8 + 0.0156 * Re_ ** 0.85 * Pr_ ** 0.93
        if Pr_ < 0.1:
            Nusselt = Nusselt_lp
        else:
            Nusselt = Nusselt_lp + (Pr_ - 0.1) * (Nusselt - Nusselt_lp) / 0.4   
            
    # Correct f and Nusselt for developing flow
    f = friction * (1 + (1 / L_ID)**0.7)
    Nu = Nusselt * (1 + (1 / L_ID)**0.7)
    
    return Nu, f

def dP_Pipe(m_dot, fluid, pipe):
    """Calculate pressure drop for flow with heat transfer at the surface of the pipe.
    Section 5.2.3 and 5.2.4 of Nellis and Klein (2020)
    
    Parameters
    ----------
    m_dot : Quantity {mass: 1, time: -1}
        mass flow rate
    fluid : ThermState
        Inlet fluid conditions
    pipe : Pipe
        Defining the pipe characteristics 
    Returns
    -------
    Quantity {length: -1, mass: 1, time: -2}
        Pressure drop
 
        Heat_transfer_coefficients
    """

    # Calculate fluid pameters
    Re_ = Re(fluid, m_dot, pipe.ID, pipe.area)
    L_ID = pipe.L.m_as(ureg.m)/pipe.ID.m_as(ureg.m)
    eps = (pipe.eps/pipe.ID)
    w = velocity(fluid, m_dot, pipe.area)
    Pr = fluid.Prandtl
    
    #Verify two phase flow and Chockedflow
    if (fluid.phase == 0 or fluid.phase == 6) and fluid.Q < 0.9: 
        Phase = 'liquid or two-phase'
    else:
        if Mach(fluid, w) > 1/(fluid.gamma):
            raise ChokedFlow(' Reduce hydraulic resistance or mass flow.')

    # Check Flow conditions
    if Re_ < 0.001 or Re_ > 5e7: 
        raise ValueError(f'Reynolds number (Re) must be > 0.001. The value is {Re_}')
    if eps < 0 or eps > 0.05:
        raise ValueError(f'Relative roughness (eps) should be between 0 and 0.05. The value is {eps}')

    if Re_ > 3000:  # Turbulent flow
        Nu_T, f = turbulent_flow(Re_, Pr, L_ID, eps)
        Nu_Q = Nu_T
        
    elif Re_ < 2300:  # Laminar flow
        Nu_T, Nu_Q, f = laminar_flow(Re_, Pr, L_ID)
        
    else:  # Transitional flow (Re between 2300 and 3000)
        Nu_T_turbulent, f_turbulent = turbulent_flow(3000, Pr, L_ID, eps)
        Nu_lam_T, Nu_lam_Q, f_lam = laminar_flow(2300, Pr, L_ID)
        
        # Interpolate between laminar and turbulent values
        alpha = (Re_ - 2300) / (3000 - 2300)
        Nu_T = Nu_lam_T + alpha * (Nu_T_turbulent - Nu_lam_T)
        Nu_Q = Nu_lam_Q + alpha * (Nu_T_turbulent - Nu_lam_Q)
        f = f_lam + alpha * (f_turbulent - f_lam)
     
    #Pressure drop
    dP = dP_Darcy(f*L_ID, fluid.Dmass, w)    
    
    #Heat transfer
    h_T = heat_trans_coef(fluid, Nu_T, pipe.ID)
    h_Q = heat_trans_coef(fluid, Nu_Q, pipe.ID)
    
    return dP.to(ureg.pascal), h_T, h_Q

#Find the inside or outside wall temperature
def find_Tw(x, T_avg, pipe, h_coeff, m_dot):
  
    """Calculate the inlet and outlet average temperature of the wall of the component.
    
    Parameters
    ----------
    fluid : ThermState
        Inlet fluid conditions
    fluid_external : Thermstate
        External fluid conditions
    pipe : Pipe
        Defining the pipe characteristics      
    dH : Quantity {length: 2, time: -2}
        specific enthalpy of fluid
    k : Quantity {length: 1, mass: 1, temperature: -1, time: -3}
        material thermal conductivity
    dT : Quantity {temperature: 1}
        temperature difference
    Tw_i: Quantity {temperature: 1}
        inside temperature of wall  
    Tw_o: Quantity {temperature: 1}
        outside temperature of wall         
    Tw_other: Quantity {temperature: 1}
        temperature of the "other" wall not in the function - EXPLAIN THIS BETTER
    dQ : Quantity {length: 2, mass: 1, time: -3}
        heat flow rate
    m_dot : Quantity { mass: 1, time: -1}
            mass flow rate
    Q_def : Quantity {mass: 1, time: -3}
            Heat load reaching the fluid
    h_coeff: Quantity : {mass : 1, temperature : -1, time : -3}
            heat transfer coefficient: chosed to be either h_T or h_Q
    Returns
    -------
    (dH - dQ) ** 2 : Equation
            Quadratic Expression that computes wall temperature when the minimum is solved
    """
    
    if pipe.Q_def != None:
        #For a system with defined heat load: pipe_Q_def
        Tw_i = T_avg + pipe.Q_def/h_coeff
        Tw_other = Tw_i
        
    elif pipe.Tw_def != None:
        #For a system with defined external temperature: pipe_Tw_def
        Tw_other = pipe.Tw_def
        
    elif pipe.T_ext != None:
        #For a system with defined external fluid or external heat transfer coeff: pipe_h_ext
        fluid_external = ThermState('air', T= pipe.T_ext, P=1 * ureg.bar) #to do: improve structure 
        h_ext = h_ext_(fluid_external, pipe, x * ureg.K)     
        
        Tw_i = (h_ext * pipe.OD * (pipe.T_ext - x * ureg.K) / h_coeff / pipe.ID) + T_avg
        Tw_i = max(min(Tw_i, pipe.T_ext), T_avg)
        Tw_other = Tw_i
        
    elif  pipe.isolation.T_ext != None and pipe.isolation.k != None:
        #For a defined insulated system: pipe_insulated
        Tw_i = (pipe.isolation.k * (pipe.isolation.T_ext - x * ureg.K) / h_coeff / pipe.ID / log(pipe.isolation.OD / pipe.OD)) + T_avg
        Tw_i = max(min(Tw_i, pipe.isolation.T_ext), T_avg)
        Tw_other = Tw_i
    else:
        raise ValueError("Insufficient or invalid parameters provided.")                
    
    k = k_pipe(pipe, Tw_other, x * ureg.K)
    dQ = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, (x * ureg.K - Tw_other))
    dH = -h_coeff * (T_avg - Tw_other) * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14
    return (dH - dQ).m_as(ureg.W) ** 2  
    


def pipe_Q_def(fluid, pipe, m_dot, dP, h_Q):
    
    """Calculate the inlet and outlet average temperature of the wall of the component.

    Parameters
    ----------
    fluid : ThermState
        Inlet fluid conditions
    fluid_downstream : Thermstate
        Outlet fluid conditions
    pipe : Pipe
        Defining the pipe characteristics 
        
    dH : Quantity {length: 2, time: -2}
        specific enthalpy of fluid
    k : Quantity {length: 1, mass: 1, temperature: -1, time: -3}
        material thermal conductivity
    dT : Quantity {temperature: 1}
        temperature difference
    dQ : Quantity {length: 2, mass: 1, time: -3}
         heat flow rate
    m_dot : Quantity { mass: 1, time: -1}
            mass flow rate
    Q_def : Quantity {mass: 1, time: -3}
            Heat load reaching the fluid
    Returns
    -------
    Tw_i, Tw_o : Quantity {temperature: 1}
        Inside temperature of the wall, Outside temperature of the wall
    """
    
    #Calculate downstream conditions
    fluid_downstream = fluid.copy()
    dH = (pipe.Q_def * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14) / m_dot 
    fluid_downstream.update('P', fluid.P - dP, 'Hmass' , fluid.Hmass + dH.to(ureg.J/ureg.kg)) 
    
    ##Calculate the average temperature of the fluid inside the component
    T_avg = (fluid.T + fluid_downstream.T)/2
    
    ### internal wall temperature
    # Tw_i = T_avg + pipe.Q_def/h_Q  

    ### external wall temperature 
    # def find_Tw_o_Q_def(x):
    #     k = k_pipe(pipe, Tw_i, x * ureg.K)
    #     dT = (x - Tw_i.m_as(ureg.K)) * ureg.K
    #     dQ = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, dT)
    #     return dH*m_dot - dQ     
        
    #Calculate Tw_i and Tw_o: minimum of the quadratic find_Tw_o
    Tw_i = T_avg + pipe.Q_def/h_Q
    Tw_o = minimize(find_Tw, x0=T_avg.m_as(ureg.K) + 1, args=(T_avg, pipe, h_Q, m_dot), bounds=[(1,3000)]).x[0] * ureg.K
    
    return Tw_i, Tw_o  
 

# Define main functions
def pipe_Tw_def(fluid, pipe, m_dot, dP, h_T):
    """Calculate the inlet and outlet average temperature of the wall of the component,
        as well as the heat load reaching the fluid.
    Parameters
    ----------
    fluid : ThermState
       Inlet fluid conditions
    fluid_downstream : Thermstate
       Outlet fluid conditions
    pipe : Pipe
        Defining the pipe characteristics 
   
    dH : Quantity {length: 2, time: -2}
        specific enthalpy of fluid
    k : Quantity {length: 1, mass: 1, temperature: -1, time: -3}
        material thermal conductivity
    dT : Quantity {temperature: 1}
        temperature difference
    dQ : Quantity {length: 2, mass: 1, time: -3}
         heat flow rate
    m_dot : Quantity { mass: 1, time: -1}
            mass flow rate
    h_T : Quantity : {mass : 1, temperature : -1, time : -3}
            heat transfer coefficient 
    T_ds : Quantity {temperature: 1}
            temperature of fluid downstream     
    Returns
    -------
    Tw_i and Tw_o : Quantity {temperature: 1}
        Inside temperature of the wall, Outside temperature of the wall
    Q : Quantity { mass: 1, time: -3}
        Heat load reaching the fluid
    """
    #### Initial conditions and parameters
    H = fluid.Hmass
    fluid_downstream = fluid.copy()
    T_avg = fluid.T
    Tw_o = pipe.Tw_def
    res = 1
    j = 0
    dT = 'none'
    
    while res>0.0001:  
        
        # ###Calculate Tw_i               
        # def find_Tw_i(x):
        #     k = k_pipe(pipe, Tw_o, x * ureg.K)
        #     dQ = (conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, x * ureg.K - Tw_o)).m_as(ureg.watt)
        #     dH = (h_T * (T_avg - x * ureg.K) * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14).m_as(ureg.watt)
        #     return (dH ** 2 - dQ ** 2)

        #Defines boundries for Tw_o
        if T_avg < Tw_o:
            bracket = [(T_avg.m_as(ureg.K), Tw_o.m_as(ureg.K))]   # Limits search range- 
        else:
            bracket = [(Tw_o.m_as(ureg.K), T_avg.m_as(ureg.K))]
        
        #Calculate Tw_i: minimum of the quadratic find_Tw_i
        Tw_i = minimize(find_Tw,  x0 = T_avg.m_as(ureg.K), args = (T_avg, pipe, h_T, m_dot), bounds=bracket).x[0] *ureg.K
        
        #Update downstream fluid conditions
        dT = T_avg - Tw_i
        dH = - h_T * dT * pipe.ID * pipe.L * 3.14 / m_dot
        fluid_downstream.update('P', fluid.P - dP,'Hmass', H + dH.to(ureg.J/ureg.kg))
        T_ds = fluid_downstream.T 

        ###Check convergence of T_average               
        T_avg_new = (fluid.T + T_ds)/2
        res = ((T_avg_new - T_avg)**2 / (T_ds - fluid.T)**2)
        
        ##Calculate updated average temperature
        T_avg = T_avg_new

        ### Eliminate nonphysical solutions         
        if (fluid.T < Tw_o and T_ds > Tw_o) or (fluid.T > Tw_o and T_ds < Tw_o):
            if j > 0:
                raise Exception('the pipe is too long')
            j += 1
            T_avg = (fluid.T + Tw_o) / 2
            
        ### Eliminate nonphysical solutions 
        # if fluid.T < Tw_o and T_ds > Tw_o:
        #     if j>0:
        #         raise Exception('the pipe is too long')
        #     j=j+1
        #     T_avg = (fluid.T + Tw_o) / 2
        # if fluid.T > Tw_o and T_ds < Tw_o:
        #     if j>0:
        #         raise Exception('the pipe is too long')
        #     j=j+1
        #     T_avg = (fluid.T + Tw_o) / 2
    
        Q = (- h_T * dT ).to(ureg.W/ureg.m ** 2)       
 
        return Tw_i, Tw_o, Q


def pipe_h_ext(fluid, pipe, m_dot, dP, h_T): 
    """Calculate the inlet and outlet average temperature of the wall of the component,
        as well as the heat load reaching the fluid considering a situation with an external fluid
     Parameters
     ----------
     fluid : ThermState
        Inlet fluid conditions
     fluid_external : Thermstate
        External fluid conditions
     fluid_downstream : Thermstate
         Outlet fluid conditions
     pipe : Pipe
         Defining the pipe characteristics 
    
     dH : Quantity {length: 2, time: -2}
         specific enthalpy of fluid
     k : Quantity {length: 1, mass: 1, temperature: -1, time: -3}
         material thermal conductivity
     dT : Quantity {temperature: 1}
         temperature difference
    dQ : Quantity {length: 2, mass: 1, time: -3}
         heat flow rate
     m_dot : Quantity { mass: 1, time: -1}
         mass flow rate
    T_ds : Quantity {temperature: 1}
             temperature of fluid downstream
    h_ext : Quantity : {mass : 1, temperature : -1, time : -3}
            external heat transfer coefficient
        
     Returns
     -------
     Tw_i and Tw_o : Quantity {temperature: 1}
         Inside temperature of the wall, Outside temperature of the wall
     Q : Quantity { mass: 1, time: -3}
         Heat load reaching the fluid
    """
    ### Initial conditions and parameters
    H = fluid.Hmass
    fluid_downstream = fluid.copy()
    T_avg = fluid.T
    res = 1
    j = 0
    
    while res>0.0001:  
        
        ##Define External Fluid
        fluid_external = ThermState('air', T= pipe.T_ext, P=1 * ureg.bar) 
             
        # def find_Tw_o(x):
        
        #     h_ext = h_ext_(fluid_external, pipe, x * ureg.K)
            
        #     #Assign reference temperature for Tw_i
        #     Tw_i = ( h_ext * pipe.OD * (pipe.T_ext - x * ureg.K) / h_T / pipe.ID  ) + T_avg 
            
        #     #Adjust reference temperature for extreme cases
        #     if Tw_i > pipe.T_ext:
        #         Tw_i = pipe.T_ext
        #     if Tw_i < T_avg:
        #         Tw_i = T_avg
                
        #     # Tw_i = max(min(T1, pipe.T_ext), T_avg)
        #     k = k_pipe(pipe, Tw_i, x * ureg.K)
           
        #     # dT1 = x * ureg.K - Tw_i
        #     # dQ = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, dT1).m_as(ureg.watt)
        #     dQ = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, (x * ureg.K - Tw_i)).m_as(ureg.watt)
           
        #     # dT2 = T_avg - Tw_i
        #     # dH = - (h_T * dT2 * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14).m_as(ureg.watt)
        #     dH = - (h_T * (T_avg - Tw_i) * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14).m_as(ureg.watt)
        #     return ((dH **2) - (dQ ** 2))**2     

        
        # Assign Range for Tw_o
        if T_avg < pipe.T_ext:   
            bracket = [(T_avg.m_as(ureg.K)+0.0001, pipe.T_ext.m_as(ureg.K)-0.0001)]   # Limits search range- 
        else:
            bracket = [(pipe.T_ext.m_as(ureg.K)+0.0001, T_avg.m_as(ureg.K)-0.0001)]       
        
        #Calculate Tw_o: minimum of the quadratic find_Tw_o
        Tw_o = minimize(find_Tw, x0=(fluid_external.T + T_avg).m_as(ureg.K)/ 2, args = (T_avg, pipe, h_T, m_dot), bounds=bracket).x[0] * ureg.K                                                  
        
        #Calculate flow parameters
        h_ext = h_ext_(fluid_external, pipe, Tw_o)
        Tw_i = ( h_ext * pipe.OD * (pipe.T_ext - Tw_o) / h_T / pipe.ID  ) + T_avg
        
        ###Calculate downstream flow conditions
        dT = T_avg - Tw_i
        dH = - h_T * dT * pipe.ID * pipe.L * 3.14 / m_dot
        fluid_downstream.update('P', fluid.P - dP,'Hmass', H + dH.to(ureg.J/ureg.kg))
        T_ds = fluid_downstream.T  

        ###Check convergence of T_average and update T_avg value            
        T_avg_new = (fluid.T + T_ds)/2
        res = ((T_avg_new - T_avg)**2 / (T_ds - fluid.T)**2)
        T_avg = T_avg_new
    
        ### Eliminate nonphysical solutions         
        if (fluid.T < Tw_o and T_ds > Tw_o) or (fluid.T > Tw_o and T_ds < Tw_o):
            if j > 0:
                raise Exception('the pipe is too long')
            j += 1
            T_avg = (fluid.T + Tw_o) / 2
         
        Q = (- h_T * dT ).to(ureg.W/ureg.m ** 2)
    
        return Tw_i, Tw_o, Q

def pipe_insulated(fluid, pipe, m_dot, dP, h_T): 
    """Calculate the inlet and outlet average temperature of the wall of the component,
        as well as the heat load reaching the fluid.
     Parameters
     ----------
        fluid : ThermState
        Inlet fluid conditions
     fluid_external : Thermstate
        External fluid conditions
     fluid_downstream : Thermstate
         Outlet fluid conditions
     pipe : Pipe
         Defining the pipe characteristics 
    
     dH : Quantity {length: 2, time: -2}
         specific enthalpy of fluid
     k : Quantity {length: 1, mass: 1, temperature: -1, time: -3}
         material thermal conductivity
     dT : Quantity {temperature: 1}
         temperature difference
    dQ : Quantity {length: 2, mass: 1, time: -3}
         heat flow rate
     m_dot : Quantity { mass: 1, time: -1}
         mass flow rate
    T_ds : Quantity {temperature: 1}
             temperature of fluid downstream  
    h_ext : Quantity : {mass : 1, temperature : -1, time : -3}
            external heat transfer coefficient
        
     Returns
     -------
     Tw_i and Tw_o : Quantity {temperature: 1}
         Inside temperature of the wall, Outside temperature of the wall
     Q : Quantity { mass: 1, time: -3}
         Heat load reaching the fluid
    """
    ###Initial conditions and parameters
    H = fluid.Hmass
    fluid_downstream = fluid.copy()
    T_avg = fluid.T
    res = 1
    j = 0
    
    while res>0.0001:   
        
        # def find_Tw_o(x):
        #     Tw_i = ( pipe.isolation.k * (pipe.isolation.T_ext - x * ureg.K) / h_T / pipe.ID / log(pipe.isolation.OD/pipe.OD) ) + T_avg
        #     if Tw_i > pipe.isolation.T_ext:
        #         Tw_i = pipe.isolation.T_ext
        #     if Tw_i < T_avg:
        #         Tw_i = T_avg
                
        #     # Tw_i = max(min(Tw_i, pipe.isolation.T_ext), T_avg)
        #     k = k_pipe(pipe, Tw_i, x * ureg.K)
        #     # dT1 = x * ureg.K - Tw_i
        #     # dQ = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, dT1)
        #     dQ = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, (x * ureg.K - Tw_i))
        #     # dT2 = T_avg - Tw_i
        #     # dH = - h_T * dT2 * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14
        #     dH = - h_T * (T_avg - Tw_i) * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14
        #     return (dH - dQ)**2
        
        # fact = (pipe.isolation.OD.m_as(ureg.m) - pipe.OD.m_as(ureg.m)) / pipe.isolation.k.m_as(ureg.W/ureg.m/ureg.K)
        # fact = min(1/2, fact)  # Use min for the upper limit      
        # Tw_o = minimize(find_Tw_o, x0=pipe.isolation.T_ext.m_as(ureg.K) - (pipe.isolation.T_ext.m_as(ureg.K) - T_avg.m_as(ureg.K)) * fact, bounds=[(T_avg.m_as(ureg.K), pipe.isolation.T_ext.m_as(ureg.K))]).x[0] * ureg.K
        # Assign Range for Tw_o
               
        if T_avg < pipe.isolation.T_ext:  
            bracket = [(T_avg.m_as(ureg.K)+0.0001, pipe.isolation.T_ext.m_as(ureg.K)-0.0001)]  # Limits search range- 
        else:
            bracket = [(pipe.isolation.T_ext.m_as(ureg.K)+0.0001, T_avg.m_as(ureg.K)-0.0001)]
         
        #Calculate Tw_o: minimum of the quadratic find_Tw_o and calculate Tw_i
        Tw_o = minimize(find_Tw, x0=pipe.isolation.T_ext.m_as(ureg.K) - (pipe.isolation.T_ext.m_as(ureg.K) - T_avg.m_as(ureg.K)), args = (T_avg, pipe, h_T, m_dot), bounds=bracket).x[0] * ureg.K 
        Tw_i = (( pipe.isolation.k * (pipe.isolation.T_ext - Tw_o) / h_T / pipe.ID / log(pipe.isolation.OD/pipe.OD) ) + T_avg)
        
        ###Calculate downstream flow conditions
        dT = T_avg - Tw_i
        dH = - h_T * dT * pipe.ID * pipe.L * 3.14 / m_dot
        fluid_downstream.update('P', fluid.P - dP,'Hmass', H + dH.to(ureg.J/ureg.kg))
        T_ds = fluid_downstream.T 

        ###Check convergence of T_average and update T_avg value       
        T_avg_new = (fluid.T + T_ds)/2
        res = ((T_avg_new - T_avg)**2 / (T_ds - fluid.T)**2)
        T_avg = T_avg_new
        
        ###Eliminate nonphysical solutions
        if (fluid.T < Tw_o and T_ds > Tw_o) or (fluid.T > Tw_o and T_ds < Tw_o):
            if j > 0:
                raise Exception('the pipe is too long')
            j += 1
            T_avg = (fluid.T + Tw_o) / 2
         
        Q = (- h_T * dT ).to(ureg.W/ureg.m ** 2)

        return Tw_i, Tw_o, Q


def pipe_heat(pipe, fluid, m_dot):
    """Determine the heated status of the piping component
    """

    ### Calculate pressure drop and heat transfer coefficient
    dP, h_T, h_Q = dP_Pipe(m_dot, fluid, pipe)  
    
    ###heat flux defined on the wall of the pipe
    if hasattr(pipe, 'Q_def') and pipe.Q_def != None :
        try: 
            pipe.Q_def.m_as(ureg.W / ureg.m ** 2)
        except:
            raise ValueError(f"the Q_def is not properly defined in component {pipe}" )
        Tw_i, Tw_o = pipe_Q_def(fluid, pipe, m_dot, dP, h_Q)
        Q = pipe.Q_def
        
    ###Temperature defined on the wall of the pipe
    elif hasattr(pipe, 'Tw_def') and pipe.Tw_def != None :
        try: 
            pipe.Tw_def.m_as(ureg.K)
        except:
            raise ValueError(f"the Tw_def is not properly defined in component {pipe}" )
        Tw_i, Tw_o, Q = pipe_Tw_def(fluid, pipe, m_dot, dP, h_Q)


    
    # ###Heat transfer defined on the wall of the pipe
    # elif hasattr(pipe, 'h_ext') and pipe.h_ext != None :
    #     try: 
    #         pipe.h_ext.m_as(ureg.W / ureg.m ** 2 / ureg.K)
    #     except:
    #         raise ValueError(f"the h_ext is not properly defined in component {pipe}" )
    #     try:
    #         pipe.T_ext.m_as(ureg.K)
    #     except:
    #         pipe.T_ext = 293 * ureg.K
    #     Tw_i, Tw_o, Q = pipe_h_ext(fluid, pipe, m_dot, dP, h_T)
        
    # ###Ambiant/external temperature defined 
    # elif hasattr(pipe, 'h_type') and pipe.h_type != None :
    #     try:
    #         pipe.safety_fact 
    #     except:
    #         pipe.safety_fact = 1
    #     Tw_i, Tw_o, Q = pipe_h_ext(fluid, pipe, m_dot, dP, h_T)
    
    
        ### Heat transfer or ambient/external temperature defined on the wall of the pipe -NEW^^
    elif (hasattr(pipe, 'h_ext') and pipe.h_ext is not None): #or (hasattr(pipe, 'h_type') and pipe.h_type is not None):
        if hasattr(pipe, 'h_ext') and pipe.h_ext is not None:
            try:
                pipe.h_ext.m_as(ureg.W / ureg.m ** 2 / ureg.K)
            except:
                raise ValueError(f"the h_ext is not properly defined in component {pipe}")
    
            try:
                pipe.T_ext.m_as(ureg.K)
            except:
                pipe.T_ext = 293 * ureg.K
    
        # if hasattr(pipe, 'h_type') and pipe.h_type is not None:
        #     try:
        #         pipe.safety_fact
        #     except:
        #         pipe.safety_fact = 1
    
        Tw_i, Tw_o, Q = pipe_h_ext(fluid, pipe, m_dot, dP, h_T)  

    ###Isolation on the external of the pipe 
    elif hasattr(pipe, 'isolation') and pipe.isolation != None :
        try: 
            pipe.isolation.k.m_as(ureg.W / ureg.m / ureg.K)
            pipe.isolation.OD.m_as(ureg.m)
        except:
            raise ValueError(f"the isolation (k, OD) is not properly defined in component {pipe}" )
        Tw_i, Tw_o, Q = pipe_insulated(fluid, pipe, m_dot, dP, h_T)
                
    ###Other
    else: 
        pipe.Q_def = Q = 0 * ureg.W/ ureg.m ** 2
        Tw_i, Tw_o = pipe_Q_def(fluid, pipe, m_dot, dP, h_Q)
    
    return Tw_i.to(ureg.K), Tw_o.to(ureg.K), dP.to(ureg.bar), Q.to(ureg.W/ ureg.m ** 2)
        

def k_pipe(pipe, T_wall, T_ext=293 * ureg.K):     ### you should add the possibility to define a table of values to do that (for materials not in the database)
    try:
        # Check if the 'pipe' object has the attribute 'k'
        if hasattr(pipe, 'k'):
            k = pipe.k.m_as(ureg.W / ureg.K / ureg.m) * ureg.W / ureg.K / ureg.m
        else:
            raise AttributeError("Pipe object has no attribute 'k'")
    except AttributeError:
        # Handle the case where 'k' is not an attribute of 'pipe'
        try:
            mat = pipe.material
        except AttributeError:
            # If 'pipe' has no 'material' attribute, default to a material
            if isinstance(pipe, CopperTube):
                mat = Material.OFHC
            else:
                mat = Material.SS304
       ### Determine the thermal conductivity using the 'nist_property' function
        if np.abs((T_wall - T_ext).m_as(ureg.K)) < 0.001:
            k = nist_property(mat, Property.TC, T1=T_wall)
        else:
            if T_wall < T_ext:
                k = nist_property(mat, Property.TC, T1=T_wall, T2=T_ext)
            else:
                k = nist_property(mat, Property.TC, T1=T_ext, T2=T_wall)
    return k.to(ureg.W / ureg.K / ureg.m)

def h_ext_(fluid, pipe, T_wall):  ### you should add the possibility to define a table of values to do that
    try:
        h = pipe.h_ext.m_as(ureg.W / ureg.K / ureg.m ** 2) * ureg.W / ureg.K / ureg.m ** 2
    except:
        try:
            type = pipe.h_type
        except:   
            raise Exception('You should define an external heat transfer type pipe.h_type') #or define type = 1
        if type >= 1:
            try:
                orientation = pipe.orientation
            except:
                orientation = None
            Ra_ = Ra(fluid, T_wall, pipe.OD)
            Pr_ = Pr(fluid)
            if orientation == 'vertical':        
                Nu_ = Nu_vcyl(Pr_, Ra_, pipe.OD, pipe.L)
            elif orientation == 'horizontal':
                Nu_ = Nu_hcyl(Pr_, Ra_)
            else:
                Nu_ = max (Nu_vcyl(Pr_, Ra_, pipe.OD, pipe.L), Nu_hcyl(Pr_, Ra_))
            h = heat_trans_coef(fluid, Nu_, pipe.OD)
        if type == 2:
            try:
                epsilon = pipe.epsilon
            except:
                epsilon = 0.075 # polished stainless steel emissivity
            sigma = 5.670373e-8 * ureg.W/ureg.m ** 2 / ureg.K ** 4
            h_rad = epsilon * sigma * (fluid.T ** 4 - T_wall ** 4) / (fluid.T - T_wall) 
            h = h + h_rad

        if type == 3: ###including Rad ice and h_ice specific in the problem to do 
            print('to do')
    return h.to(ureg.W / ureg.K / ureg.m ** 2)