# script for solving the heat equation numerically
import numpy as np
import matplotlib.pyplot as plt

def analytical_sol(x, t, L):
    return np.sin(np.pi*x/L)*np.exp(-(np.pi/L)**2*t)

def Ahat(Nx, alpha):
    A = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i == j:
                A[i, j] = 1 - 2*alpha
            if i == j-1 or i == j+1:
                A[i,j] = alpha
    return A

def pde_iter(Vj, A):
    Vjp1 = A @ Vj
    return Vjp1

def solve_pde_explicit_euler(V0, dx, dt, Nx, Nt):
    alpha = dt/dx**2
    print("alpha = ", alpha)
    assert alpha <= 1/2 # from spectral radius requirement for stability
    shapeVs = tuple([Nt,] + list(V0.shape) )
    V = np.ones(shapeVs)*np.nan
    V[0] = V0
    V[:,0] = 0
    V[:,-1] = 0
    A = Ahat(Nx-2, alpha)
    for j in range(1,Nt):
        V[j,1:-1] = pde_iter(V[j-1,1:-1], A)
        l2norm_consec = np.sum((V[j]-V[j-1])**2)**.5
        if j % 100 == 0: print("l2norm of difference between consecutive iterations: ", l2norm_consec)
    return V, l2norm_consec

if __name__ == '__main__':
    alpha = 0.4
    dxs = [5e-3,1e-2,5e-2,1e-1]
    dts = [dx**2*alpha for dx in dxs]
    errors = []
    # Slices at different times
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle("Numerical solution for $t \\in \{ T/4, TL/4 \}$")
    # BCs
    fig0, ax0 = plt.subplots(1,3,figsize=(10,5))
    fig0.suptitle("Numerical solution for $t = 0$  $x \\in \{ 0, L \}$")

    for i in range(len(dxs)):
        dx = dxs[i]
        dt = dts[i]
        L = 1
        T = 1
        Nt = int(np.round(L/dt))+1
        Nx = int(np.round(T/dx))+1
        print("Nt, Nx", Nt, Nx)
        x = np.linspace(0,L, Nx)
        t = np.linspace(0,T, Nt)
        assert np.isclose(dt, t[1]-t[0], atol=1e-10), print("dt and t[1]-t[0]", dt, t[1]-t[0])
        assert np.isclose(dx, x[1]-x[0], atol=1e-10), print("dx and x[1]-x[0]", dx, x[1]-x[0])
        xmesh, tmesh = np.meshgrid(x, t)
        V0 = np.sin(np.pi*x/L).reshape(-1,1)
        V, l2_last = solve_pde_explicit_euler(V0, dx, dt, Nx, Nt)
        Vanal = analytical_sol(xmesh, tmesh, L)
        eps_mean = np.mean((V[:,:,0]-Vanal)**2)**.5
        errors.append(eps_mean)
        plotsurf=False
        if plotsurf:
            fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
            ax.plot_surface(xmesh, tmesh, V[:,:,0], cmap=plt.cm.coolwarm)
            ax.plot_surface(xmesh, tmesh, Vanal,    cmap=plt.cm.spring, alpha=0.4)
            plt.show()
        # exponential regime
        ax[0].plot(x, V[Nt//4,:,0], label='numerical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ) )
        ax[0].plot(x, Vanal[Nt//4,:], label='analytical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        # linear regime
        ax[1].plot(x, V[3*Nt//4,:,0], label='numerical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        ax[1].plot(x, Vanal[3*Nt//4,:], label='analytical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        ax[1].legend(bbox_to_anchor=(1.,1.,0.,0.))
        fig.tight_layout()
        # Dirichlet BCs
        # at t=0
        ax0[0].plot(x, V[0,:,0], label='numerical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ) )
        ax0[0].plot(x, Vanal[0,:], label='analytical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        # at x=0
        ax0[1].plot(t, V[:,0,0], label='numerical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        ax0[1].plot(t, Vanal[:,0], label='analytical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        # at x=L
        ax0[2].plot(t, V[:,Nx-1,0], label='numerical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        ax0[2].plot(t, Vanal[:,Nx-1], label='analytical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        ax0[2].legend(bbox_to_anchor=(1.,1.,0.,0.))
        fig0.tight_layout()

    plt.show()
    fig, ax = plt.subplots(1,2,figsize=(10,5), sharey=True)
    fig.suptitle("Numerical accuracy in $\\Delta t$ and $\\Delta x$")
    dtlin = np.logspace(np.log10(min(dts)), np.log10(max(dts)),10)
    dxlin = np.logspace(np.log10(min(dxs)), np.log10(max(dxs)),10)
    ax[0].plot(dxs, errors, label="$\\epsilon$", ls='--', marker = 'o')
    ax[0].plot(dxlin, dxlin**2, label="$\\Delta x^2$")
    ax[1].plot(dts, errors, label="$\\epsilon$", ls='--', marker = 'o')
    ax[1].plot(dtlin, dtlin, label="$\\Delta t$")
    for a in ax:
        a.set_xscale('log')
        a.set_yscale('log')
        a.legend()
    plt.show()

    
