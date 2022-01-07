

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
l1 = 2.0
l2 = 2.0
g = 9.8
def initial(orbit_noise=5e-2, min_radius=0.5, max_radius=1.5):
    global g
    state = np.zeros((2,3))
    state[:,0] = [1, 1]  #m1,  m2
    # get initial state
    state[:,1] = (np.random.rand(2)-0.5)*np.pi #theta, phi
    state[:,2] = (np.random.rand(2)-0.5)*np.pi/2 #p_theta ,p_phi
    return state

def hamiltonian_fn(coords):
    global l1, l2, g
    state[:,1], state[:,2]= np.split(coords,2)
    H = (state[1,0]*(l2*state[0,2])**2+(state[0,0]+state[1,0])*(l1*state[1,2])**2-2*state[1,0]*l1*l2*state[0,2]*state[1,2]*np.cos(state[0,1]-state[1,1]))/(2*state[1,0]*((l1*l2)**2)*(state[0,0]+state[1,0]*(np.sin(state[0,1]-state[1,1]))**2))
    -(state[0,0]+state[1,0])*g*l1*np.cos(state[0,1])-state[1,0]*g*l2*np.cos(state[1,1])# pendulum hamiltonian
    return H

def dynamics_fn(t,
                coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(state, t_span=[0,3], timescale=15, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=np.concatenate([state[:1], state[:2]]), t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][:0], spring_ivp['y'][:1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
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
random_config(orbit_noise=5e-2, min_radius=0.5, max_radius=1.5)
print(1)
