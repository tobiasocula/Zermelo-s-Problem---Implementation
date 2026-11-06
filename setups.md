k = 2.0
c = 0.5
A, B = 2, 2
u = lambda t,x,y: np.zeros_like(x)
v = lambda t,x,y: -c*y
u_x = lambda t,x,y: np.zeros_like(x)
u_y = lambda t,x,y: np.zeros_like(x)
v_x = lambda t,x,y: np.zeros_like(x)
v_y = lambda t,x,y: -c*np.ones_like(y)

c = 3
u = lambda t,x,y: -y
v = lambda t,x,y: 0.0
k = 1
A, B = 0, 0.5
u_x = lambda t,x,y: np.zeros_like(x)
u_y = lambda t,x,y: -np.ones_like(y)
v_x = lambda t,x,y: np.zeros_like(x)
v_y = lambda t,x,y: np.zeros_like(y)

c = 6
k = 10
A, B = 10, 10
L = 10.0          # length scale of the current pattern
w = np.pi / L
u   = lambda t, x, y: -c * np.sin(w * y)
v   = lambda t, x, y:  c * np.sin(w * x)
u_x = lambda t, x, y: np.zeros_like(x)
u_y = lambda t, x, y: -c * w * np.cos(w * y)
v_x = lambda t, x, y:  c * w * np.cos(w * x)
v_y = lambda t, x, y: np.zeros_like(x)

k = 1.0
A, B = 2, 2
a, b = 0.5, 0.7
u = lambda t,x,y: a*x
v = lambda t,x,y: b*y
u_x = lambda t,x,y: np.ones_like(x)*a
u_y = lambda t,x,y: np.zeros_like(y)
v_x = lambda t,x,y: np.zeros_like(x)
v_y = lambda t,x,y: np.ones_like(y)*b

k = 1.0
A, B = 2, 2
u = lambda t,x,y: -y**2
v = lambda t,x,y: 0.0
u_x = lambda t,x,y: np.zeros_like(x)
u_y = lambda t,x,y: -2*y
v_x = lambda t,x,y: np.zeros_like(x)
v_y = lambda t,x,y: np.zeros_like(x)

k = 1.0
A, B = 2, 2
u = lambda t,x,y: -y**2
v = lambda t,x,y: 0.0
u_x = lambda t,x,y: np.zeros_like(x)
u_y = lambda t,x,y: -2*y
v_x = lambda t,x,y: np.zeros_like(x)
v_y = lambda t,x,y: np.zeros_like(x)

k = 1.0
A, B = 2, 2
u = lambda t,x,y: -t*y
v = lambda t,x,y: 0.0
u_x = lambda t,x,y: 0.0
u_y = lambda t,x,y: -t
u_y = lambda t,x,y: 0.0
v_x = lambda t,x,y: 0.0
v_y = lambda t,x,y: 0.0

