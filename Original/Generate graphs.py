import datetime
print(datetime.datetime.now())

#example - improve rational parametrization
S = MinimalSurface_square(a = [0, 0, 1, 0], b = [0, 0, 0, 1])
print("First surface"); MS.plot()
lam1, ev, B = MS.compute_maxEig()
print(lam1)

MS.improve_coeffs(step=100)
lam2, ev, B = MS.compute_maxEig()
print("Second surface"); MS.plot()

MS.improve_coeffs(step=100)
lam3, ev, B = MS.compute_maxEig()

print("Third surface"); MS.plot()
print("The eigenvalue starts with", -lam1, "and becomes", -lam2, "and then", -lam3)
print(datetime.datetime.now())
