
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
m =  1.0
k =  1.0
l0 = 1.0
g = 9.8

def hamiltonian_fn(coords):
    state = coords #r,theta,pr,ptheta
    H =1/2/m*(state[2]**2+state[3]**2/state[0]**2)+1/2*k*(state[0]-l0)**2-m*g*state[0]*np.cos(state[1])

    return H


def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dHdp, dHdq = dcoords[2:], -dcoords[0:2]
    S = np.concatenate([dHdp, dHdq], axis=-1)
    return S


def get_trajectory(t_span=[0, 20], timescale=50, noise_std=0.01, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))
    y0 = np.array([l0+0.1*np.random.rand(),np.random.rand()*np.pi/2, np.random.rand(), np.random.rand()])
    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q1, q2, p1, p2 = spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2], spring_ivp['y'][3]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dq1, dq2, dp1, dp2 = np.split(dydt, 4)

    # add noise

    q1 += np.random.randn(*q1.shape)*noise_std
    q2 += np.random.randn(*q2.shape) * noise_std
    p1 += np.random.randn(*p1.shape)*noise_std
    p2 += np.random.randn(*p2.shape) * noise_std

    return q1, q2, p1, p2, dq1, dq2, dp1, dp2, t_eval


def get_dataset(seed=0, samples=100, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        q1, q2, p1, p2, dq1, dq2, dp1, dp2, t = get_trajectory(**kwargs)
        xs.append(np.stack([q1, q2, p1, p2]).T)
        dxs.append(np.stack([dq1, dq2, dp1, dp2]).T)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data
    '''
    for s in range(samples):
        l, theta, dl, dtheta, t = get_trajectory(**kwargs)
        x, y, dx, dy=np.zeros([2,1])
        x[0,0]=l[0]*np.sin(theta[0])
        y[0,0]=-l[0]*np.cos(theta[0])
        x[1,0]=x[0,0]+l[1]*np.sin(theta[1])
        y[1,0]=y[0,0]-l[1]*np.cos(theta[1])
        dx[0,0]=dtheta[0]*np.cos(theta[0])
        dy[0,0]=dtheta[0]*l[0]*np.sin(theta[0])
        dx[1,0]=dx[0,0]+dtheta[1]*np.cos(theta[1])
        dy[1,0]=dy[0,0]+dtheta[1]*l[1]*np.sin(theta[1])

        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data
    '''


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])

    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field
