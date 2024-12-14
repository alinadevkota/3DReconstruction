import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_view_matrix(C, T, U):
    # Step 1: Compute the forward vector (f), normalize it
    f = T - C
    f = normalize(f)

    # Step 2: Compute the right vector (r), normalize it
    r = np.cross(U, f)
    r = normalize(r)

    # Step 3: Compute the true up vector (u), normalize it
    u = np.cross(f, r)
    u = normalize(u)

    # Step 4: Compute the translation (negative camera position)
    translation = np.array([-np.dot(r, C), -np.dot(u, C), np.dot(f, C)])

    # Step 5: Construct the view matrix
    view_matrix = np.array(
        [
            [r[0], u[0], -f[0], 0],
            [r[1], u[1], -f[1], 0],
            [r[2], u[2], -f[2], 0],
            [translation[0], translation[1], translation[2], 1],
        ]
    )

    return view_matrix
