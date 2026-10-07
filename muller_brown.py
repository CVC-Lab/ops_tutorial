import numpy as np
from functools import partial
import openpathsampling.engines.toy as toys

def _phi(a:float, b:float, c: float, x0:float, y0:float, x:float, y:float) -> float:
    return a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2

def _e(A: float, a:float, b:float, c:float, x0:float, y0:float, x:float, y:float) -> float:
    return A * np.exp(_phi(a, b, c, x0, y0, x, y))

def _dphi_dx(a:float, b:float, x0:float, y0:float, x:float, y:float) -> float:
    return 2 * a * (x - x0) + b * (y - y0)

def _dphi_dy(b:float, c:float, x0:float, y0:float, x:float, y:float) -> float:
    return b * (x - x0) + 2 * c * (y - y0)

def _d2phi_dx2(a:float) -> float:
    return 2 * a

def _d2phi_dxdy(b:float) -> float:
    return b

def _d2phi_dy2(c:float) -> float:
    return 2 * c

def _de_dx(A: float, a:float, b:float, c:float, x0:float, y0:float, x:float, y:float) -> float:
    return _e(A, a, b, c, x0, y0, x, y) * _dphi_dx(a, b, x0, y0, x, y)

def _de_dy(A: float, a:float, b:float, c:float, x0:float, y0:float, x:float, y:float) -> float:
    return _e(A, a, b, c, x0, y0, x, y) * _dphi_dy(b, c, x0, y0, x, y)

class MullerBrownPES(toys.PES):
    """
    Müller–Brown-style 2D potential matching

        U(x, y) = beta * (e1 + e2 + e3 + e4)

    where
        e1 = -200 * exp(-(x - 1)^2 - 10 y^2)
        e2 = -100 * exp(-x^2 - 10 (y - 0.5)^2)
        e3 = -170 * exp(-6.5 (x + 0.5)^2 + 11 (x + 0.5)(y - 1.5) - 6.5 (y - 1.5)^2)
        e4 =  15  * exp( 0.7 (x + 1)^2 + 0.6 (x + 1)(y - 1) + 0.7 (y - 1)^2)
    """

    def __init__(self, beta=1.0):
        super(MullerBrownPES, self).__init__()
        self.beta = beta
        self.A_s = [-200.0, -100.0, -170.0, 15.0]
        self.a_s = [-1.0, -1.0, -6.5, 0.7]
        self.b_s = [0.0, 0.0, 11.0, 0.6]
        self.c_s = [-10.0, -10.0, -6.5, 0.7]
        self.x0_s = [1.0, 0.0, -0.5, -1.0]
        self.y0_s = [0.0, 0.5, 1.5, 1.0]

    def V(self, sys):
        # sys is typically a 2-vector [x, y] in the toy engine
        x, y = sys.positions[0], sys.positions[1]
        for i in range(4):
            A = self.A_s[i]
            a = self.a_s[i]
            b = self.b_s[i]
            c = self.c_s[i]
            x0 = self.x0_s[i]
            y0 = self.y0_s[i]
            e_i = _e(A, a, b, c, x0, y0, x, y)
            if i == 0:
                total_e = e_i
            else:
                total_e += e_i

        return self.beta * total_e

    def dVdx(self, sys):
        """
        Return gradient dV/dx = [dV/dx, dV/dy].
        In OPS toy PES docs, dVdx is the derivative of the potential.
        """
        x, y = sys.positions[0], sys.positions[1] # np.asarray(sys)

        dV_dx = 0.0
        dV_dy = 0.0
        for i in range(4):
            A = self.A_s[i]
            a = self.a_s[i]
            b = self.b_s[i]
            c = self.c_s[i]
            x0 = self.x0_s[i]
            y0 = self.y0_s[i]
            dV_dx += _de_dx(A, a, b, c, x0, y0, x, y)
            dV_dy += _de_dy(A, a, b, c, x0, y0, x, y)
        return np.array([self.beta * dV_dx, self.beta * dV_dy])


    def Hess(self, sys):
        """
        Return Hessian matrix of second derivatives of the potential.
        In OPS toy PES docs, Hess is the second derivative of the potential.
        """
        x, y = sys.positions[0], sys.positions[1]

        d2V_dx2 = 0.0
        d2V_dxdy = 0.0
        d2V_dy2 = 0.0
        for i in range(4):
            A = self.A_s[i]
            a = self.a_s[i]
            b = self.b_s[i]
            c = self.c_s[i]
            x0 = self.x0_s[i]
            y0 = self.y0_s[i]

            e_i = _e(A, a, b, c, x0, y0, x, y)
            dphi_dx_i = _dphi_dx(a, b, x0, y0, x, y)
            dphi_dy_i = _dphi_dy(b, c, x0, y0, x, y)
            d2phi_dx2_i = _d2phi_dx2(a)
            d2phi_dxdy_i = _d2phi_dxdy(b)
            d2phi_dy2_i = _d2phi_dy2(c)

            d2V_dx2 += e_i * (dphi_dx_i ** 2 + d2phi_dx2_i)
            d2V_dxdy += e_i * (dphi_dx_i * dphi_dy_i + d2phi_dxdy_i)
            d2V_dy2 += e_i * (dphi_dy_i ** 2 + d2phi_dy2_i)
        return self.beta * np.array([[d2V_dx2, d2V_dxdy], [d2V_dxdy, d2V_dy2]])
