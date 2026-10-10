"""Periodic driver envelope; extracted unchanged."""
from __future__ import annotations

import numpy as np
from .params import njit


@njit(cache=False)
def _driver_envelope(caps,curvature,cda,cla,ds,mass,mu,rho,rolling,ratio,ice_kw,ice_force):
    speed=caps.copy();n=len(speed);residual=1e100;converged=False
    for iteration in range(40):
        old=speed.copy()
        for i in range(n-1,-1,-1):
            j=(i+1)%n;lo=.5;hi=speed[i]
            mean=.5*(speed[i]+speed[j])
            total=mu*(9.81+.5*rho*cla[i]*speed[i]*speed[i]/mass)
            lateral=speed[i]*speed[i]*abs(curvature[i])
            tire=mass*np.sqrt(max(0.,total*total-lateral*lateral))*ratio
            drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
            wheel=mass*(speed[j]*speed[j]-speed[i]*speed[i])/(2*ds)+drag
            if wheel>=-tire:continue
            for k in range(28):
                trial=.5*(lo+hi);mean=.5*(trial+speed[j])
                total=mu*(9.81+.5*rho*cla[i]*trial*trial/mass)
                lateral=trial*trial*abs(curvature[i])
                tire=mass*np.sqrt(max(0.,total*total-lateral*lateral))*ratio
                drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
                wheel=mass*(speed[j]*speed[j]-trial*trial)/(2*ds)+drag
                if wheel>=-tire:lo=trial
                else:hi=trial
            speed[i]=min(speed[i],lo)
        for i in range(n):
            j=(i+1)%n;lo=.5;hi=speed[j]
            total=mu*(9.81+.5*rho*cla[i]*speed[i]*speed[i]/mass)
            lateral=speed[i]*speed[i]*abs(curvature[i])
            tire=mass*np.sqrt(max(0.,total*total-lateral*lateral))
            mean=.5*(speed[i]+speed[j])
            drive=min(tire,ice_force,ice_kw*1000/max(mean,.5))
            drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
            wheel=mass*(speed[j]*speed[j]-speed[i]*speed[i])/(2*ds)+drag
            if wheel<=drive:continue
            for k in range(28):
                trial=.5*(lo+hi);mean=.5*(speed[i]+trial)
                drive=min(tire,ice_force,ice_kw*1000/max(mean,.5))
                drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
                wheel=mass*(trial*trial-speed[i]*speed[i])/(2*ds)+drag
                if wheel<=drive:lo=trial
                else:hi=trial
            speed[j]=min(speed[j],lo)
        residual=np.max(np.abs(old-speed))
        if residual<1e-7:
            converged=True;break
    return speed,residual,iteration+1,converged
