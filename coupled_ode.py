#!/usr/bin/python
#==============================================================
# Implicit integrator schemes for stiff coupled ODEs.
# Containing the functions
#
#==============================================================
# required libraries
import argparse	                 # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np               # numerical routines (arrays, math functions etc)
import matplotlib.pyplot as plt  # plotting commands
import utilities as util     # for rescaleplot

import ode_integrators as odeint # contains the drivers for ODE integration
import ode_step as step          # the stepper functions

#==============================================================
# function dydx = get_dydx(x,y,dx)
#
# Calculates RHS for stiff test equation
#--------------------------------------------------------------

def get_dydx(x,y,dx):
    dydx    = np.zeros(2)
    dydx[0] =  998.0*y[0]+1998.0*y[1]
    dydx[1] = -999.0*y[0]-1999.0*y[1]
    return dydx

#==============================================================
#==============================================================
# main
# 
# parameters:
#   stepper: ODE solver (see help function below)   
#

def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("stepper",type=str,default='euler',
                        help="stepping function:\n"
                             "   euler    : Euler step\n"
                             "   rk2      : Runge-Kutta 2nd order\n"
                             "   rk4      : Runge-Kutta 4th order\n"
                             "   backeuler: implicit Euler step\n")
    args   = parser.parse_args()
    if (args.stepper == "euler"):
        fORD = step.euler
    elif (args.stepper == "rk2"):
        fORD = step.rk2
    elif (args.stepper == "rk4"):
        fORD = step.rk4
    elif (args.stepper == "rk45"):
        fORD = step.rk45
    elif (args.stepper == "backeuler"):
        fORD = step.backeuler
    else:
        raise Exception("invalid stepper %s" % (args.stepper))

    # initialization
    nstep    = 100
    x0       = 0.0
    x1       = 1.0
    y0       = np.array([1.0,0.0])
    fINT     = odeint.ode_ivp
    fRHS     = get_dydx
    fBVP     = 0

    # solve
    x,y,it   = fINT(fRHS,fORD,fBVP,x0,y0,x1,nstep)

    # Your code should show three plots in one window:
    # (1) the dependent variables y[0,:] and y[1,:] against x
    # (2) the residual y[0,:]-u and y[1,:]-v against x, see homework sheet
    # (3) the number of iterations against x.
    
    #??????????????????????????????????????????????????????????
    
    def u(x):
        return 2*np.exp(-x) - np.exp(-1000*x)
        
    def v(x):
        return -np.exp(-x) + np.exp(-1000*x)
    
    u_function = y[0,:]
    v_function = y[1,:]
    u_residual = u_function - u(x)
    v_residual = v_function - v(x)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(x, u_function, '-', x, v_function, '-')
    plt.legend(('u(x)', 'v(x)'),loc = 0)
    plt.grid(True)
    plt.show()
    
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.plot(x, u_residual, '-', x, v_residual, '-')
    plt.legend(('u(x) Residuals', 'v(x) Residuals'),loc = 0)
    plt.grid(True)
    plt.show()
    
    plt.xlabel('x')
    plt.ylabel('it')
    plt.plot(x, it, '-')
    plt.title('Number of Iterations vs. x')
    plt.grid(True)
    plt.show()

    #??????????????????????????????????????????????????????????

#==============================================================

main()