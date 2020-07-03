import math
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.linalg as lg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MinimalSurface_square:
    def __init__(self, a = [0.1, 0, 1, 0], b = [0, 0, 0, 1], nmax=4):
        """
        set basic params
        a: the coeffs of the numerator of a rational function g:C -> C
        b: coeffs or the denominator of a rational function g
        e.g., the default parameters correspond to g(z) = (0.1*z^3 + 0*z^2 + z + 0)/(0*z^3 + 0*z^2 + 0*z + 1)
        nmax: no of basis functions used to approximate eigvals of -div g grad u = lambda u
        after ibp, this become int g grad u grad v dx = lambda int u v dx
        maybe it was int grad u grad v dx  = lambda int u v / g dx
        after discr, this become Av = lambda Mv
        
        """
        #store basic data
        self.a = a
        self.b = b        
        
        #create index of basis fcts
        self.indexing_basis_fcts(nmax=nmax)
        self.N=len(self.list) #number of basis functions
        self.f = 1;
        
        #discretize computationald domain and compute differentials    
        umax = math.pi; umin = -math.pi; nu = 60; #if we change these, we may need to change
        vmax = math.pi; vmin = -math.pi; nv = 60; #the basis functions, nu: no of quadpoints
        self.du=(umax-umin)/nu
        self.dv=(vmax-vmin)/nv        
        self.u = [umin+self.du*u for u in range(nu+1)]
        self.v = [vmin+self.dv*v for v in range(nv+1)]
        self.umax = math.pi; self.umin = -math.pi;
        self.vmax = math.pi; self.vmin = -math.pi;
        
        #assemble LHS matrix (indep of a and b)
        self.A = self.assemble_matrix(ifLHS=True)
        
    def indexing_basis_fcts(self, nmax=4):
        #create a list with numbers of the form [(1,1), (2,1), (1,2), ...]
        list=[]
        k = 1
        n = 1
        while n<nmax+1:
            i=k
            j=1
            while i>j and n<nmax+1:
                list.append((i, j))
                j = j + 1
                n = n + 1
            i = 1
            j = k
            while j > i and n < nmax+1:
                list.append((i, j))
                i = i + 1
                n = n + 1
            i = k
            j = k
            if n < nmax+1:
                num=(i, j)
                list.append(num)
                n = n + 1
            k = k + 1
        self.list = list
        
    def integrateLHS(self, j, k, jj, kk): 
        """what is this?!?!"""
        cst = (-jj**2-kk**2) * (2/math.pi) * (2/math.pi) * self.du * self.dv
        fct1= [math.sin(j*x) * math.sin(jj*x) for x in self.u]
        fct2= [math.sin(k*x) * math.sin(kk*x) for x in self.v]
        f1xf2 = [y1*y2 for y1 in fct1 for y2 in fct2]
        return cst*sum(f1xf2)

    def compute_ds(self):
        a = self.a
        b = self.b
        zt= [x+1j*y for x in self.u for y in self.v]
        nu = [((((a[0]*x) + a[1]) * x) + a[2])*x +a[3] for x in zt]
        de = [((((b[0]*x) + b[1]) * x) + b[2])*x +b[3] for x in zt]
        g = [a/b if abs(b)>1.e-13 else 0. for (a,b) in zip(nu, de)]
        ds = [(1+abs(g_)**2)**2*(abs(self.f)**2) for g_ in g]
        return(ds)

    def integrateRHS(self, j, k, jj, kk, ds):
        cst = (2/math.pi) * (2/math.pi) * self.du * self.dv
        fct1= [math.sin(j*x) * math.sin(jj*x) for x in self.u]
        fct2= [math.sin(k*x) * math.sin(kk*x) for x in self.v]
        f1xf2 = [y1*y2 for y1 in fct1 for y2 in fct2]
        assert len(ds) == len(f1xf2)
        return cst*sum([a*b for (a,b) in zip(f1xf2, ds)])

    def assemble_matrix(self, ifLHS=True):
        #assemble which-hand side matrix
        lstMtrx=[]
        if not ifLHS:
            ds = self.compute_ds()
        for i in range(self.N):
            for ii in range(self.N):
                j = self.list[i][0]
                k = self.list[i][1]
                jj = self.list[ii][0]
                kk = self.list[ii][1]
                if ifLHS:
                    lstMtrx.append(self.integrateLHS(j, k ,jj, kk))
                else:
                    lstMtrx.append(self.integrateRHS(j, k ,jj, kk, ds))

        #stack entries in matrix format
        Mat = [[lstMtrx[i+j*self.N] for j in range(self.N)] for i in range(self.N)]
        return Mat

    def compute_maxEig(self):        
        B = self.assemble_matrix(ifLHS=False)#assemble RHS matrix
        w, v = lg.eig(self.A, B)
        lam_idx = w.argmax()
        lam = w[lam_idx]
        ev = v[:,lam_idx]
        return lam, ev, np.array(B)

    def compute_dlambda(self):
        a = self.a
        b = self.b
        lam, ev, B = self.compute_maxEig()
        prefactor = lam / ev.dot(B.dot(ev))

        u_ = [0 for i in self.u]
        for idx, indeces in enumerate(self.list):
            j = indeces[0]
            k = indeces[1]
            fct= [math.sin(j*x) * math.sin(k*y) for (x,y) in zip(self.u, self.v)]
            u_ = [a + ev[idx]*b for (a,b) in zip(u_, fct)]
        u2 = [a*a for a in u_]

        cst = lam * (2/math.pi) * (2/math.pi) * self.du * self.dv * prefactor
        zt= [x+1j*y for x in self.u for y in self.v]

        de = [((((b[0]*x) + b[1]) * x) + b[2])*x +b[3] for x in zt] 
        dgda1 = [x*x*x/y for (x,y) in zip(zt, de)]
        dgda2 = [  x*x/y for (x,y) in zip(zt, de)]
        dgda3 = [    x/y for (x,y) in zip(zt, de)]
        dgda4 = [    1/y for (x,y) in zip(zt, de)]

        de2 = [y*y for y in de]
        nu_ = [-((((a[0]*x) + a[1]) * x) + a[2])*x +a[3] for x in zt]
        nu_ = [a/b for (a,b) in zip(nu_,de2)]
        dgdb1 = [a*x*x*x for (a,x) in zip(nu_,zt)]
        dgdb2 = [a*x*x   for (a,x) in zip(nu_,zt)]
        dgdb3 = [a*x     for (a,x) in zip(nu_,zt)]
        dgdb4 = [a       for (a,x) in zip(nu_,zt)]

        dlada1 = cst*sum([a*b for (a,b) in zip(u2, dgda1)])
        dlada2 = cst*sum([a*b for (a,b) in zip(u2, dgda2)])
        dlada3 = cst*sum([a*b for (a,b) in zip(u2, dgda3)])
        dlada4 = cst*sum([a*b for (a,b) in zip(u2, dgda4)])
        dladb1 = cst*sum([a*b for (a,b) in zip(u2, dgdb1)])
        dladb2 = cst*sum([a*b for (a,b) in zip(u2, dgdb2)])
        dladb3 = cst*sum([a*b for (a,b) in zip(u2, dgdb3)])
        dladb4 = cst*sum([a*b for (a,b) in zip(u2, dgdb4)])

        dlambda_a = [dlada1, dlada2, dlada3, dlada4]
        dlambda_b = [dladb1, dladb2, dladb3, dladb4]
        return dlambda_a, dlambda_b
    
    def improve_coeffs(self, step=0.1):
        dlambda_a, dlambda_b = self.compute_dlambda()
        self.a = [x - step*dx for (x,dx) in zip(self.a, dlambda_a)]
        self.b = [x - step*dx for (x,dx) in zip(self.b, dlambda_b)]
    
    def improve_coeffs_numerator(self, step=0.1):
        dlambda_a, dlambda_b = self.compute_dlambda()
        self.a = [x - step*dx for (x,dx) in zip(self.a, dlambda_a)]
        
    def plot(self):
        %matplotlib inline
        #%matplotlib notebook
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt        
        
        #coefficients of rational function
        a = self.a
        b = self.b
        
        #sampling grid
        npts = 20
        tu = np.linspace(self.umin, self.umax, npts)
        tv = np.linspace(self.vmin, self.vmax, npts)
        
        #pre-allocate space
        x = [np.nan for x in range(npts**2)]
        y = [np.nan for x in range(npts**2)]
        z = [np.nan for x in range(npts**2)]
        
        #which z0 should we use? why?
        z0 = [1, 0] #assume self.f = 1
        
        for idu_, u_ in enumerate(tu):
            for idv_, v_ in enumerate(tv):
                
                nq = 100 #number of quadrature points
                uq_ = np.linspace(z0[0], u_, nq)
                vq_ = np.linspace(z0[1], v_, nq)
                zq_ = uq_ + 1j*vq_

                nu = [((((a[0]*x) + a[1]) * x) + a[2])*x +a[3] for x in zq_]
                de = [((((b[0]*x) + b[1]) * x) + b[2])*x +b[3] for x in zq_]
                g = [a/b if abs(b)>1.e-13 else 0. for (a,b) in zip(nu, de)]
                g2 = [g_*g_ for g_ in g]
                f = 1 #may change in the future

                x[npts*(idu_-1) + idv_] =  np.real(np.trapz([      (1-g2_)*f for g2_ in g2], x=zq_))
                y[npts*(idu_-1) + idv_] =  np.real(np.trapz([(1j + 1j*g2_)*f for g2_ in g2], x=zq_))
                z[npts*(idu_-1) + idv_] =  np.real(np.trapz([       (2*g_)*f for  g_ in  g], x=zq_))
        
        #import ipdb
        #ipdb.set_trace()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for ii in range(npts-1):
            for jj in range(npts-1):
                x_ = [x[npts*(ii-1) + jj], x[npts*(ii-1) + jj + 1], x[npts*(ii) + jj], x[npts*(ii) + jj + 1]]
                y_ = [y[npts*(ii-1) + jj], y[npts*(ii-1) + jj + 1], y[npts*(ii) + jj], y[npts*(ii) + jj + 1]]
                z_ = [z[npts*(ii-1) + jj], z[npts*(ii-1) + jj + 1], z[npts*(ii) + jj], z[npts*(ii) + jj + 1]]
                ax.plot_trisurf(x_, y_, z_, color=[88./256, 150./256, 212./256, 0.6], edgecolor= [0,0,0], linewidth=0.4, antialiased=True)
        #use for-loop top plot squares
        plt.show()
