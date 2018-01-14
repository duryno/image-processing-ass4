import ps_utils as ps
import numpy as np
import cv2

I, mask, S = ps.read_data_file('Buddha.mat')

x,y,z = I.shape

nz = 0
for m in mask:
    for n in m:
        if(n != 0):
            nz += 1

# create J array
J = np.zeros((10, nz))

# take image values from positions where the corresponding mask value is 1
# one image per row in J
row = 0
# J array
for j in J:
    col = 0
    i = 0
    # mask
    for m in mask:
        j = 0
        for n in m:
            if(n != 0):
                J[row,col] = I[i,j,row]
                # print(J[row,col])
                col += 1
            j += 1
        i += 1
    row += 1

# calculate the pseudo-inverse of S
s_inv = np.linalg.pinv(S)
# multiply the inverse with J
m = np.dot(s_inv, J)

# extract albedo within the mask
albedo = np.zeros((x, y, 3))
l = 0
for a in m:
    i = 0
    val = 0
    for k in mask:
        j = 0
        for n in k:
            if(n != 0):
                albedo[i,j,l] = np.linalg.norm(m[l,val])
                val += 1
            j += 1
        i += 1
    l += 1

# get albedo for export
albedo1 = np.zeros((x, y))
index = 0
for i in albedo:
    index2 = 0
    for j in i:
        albedo1[index, index2] = j[0]
        index2 += 1
    index += 1
# cv2.imwrite('images/buddha-albedo1.png', albedo1)

# extract the normal field
normal = np.zeros((x, y, 3))
l = 0
for a in m:
    i = 0
    val = 0
    for k in mask:
        j = 0
        for n in k:
            if(n != 0):
                normal[i,j,l] =  ( 1 / np.linalg.norm(m[l,val]) ) * m[l,val]
                val += 1
            j += 1
        i += 1
    l += 1

# extract normal field components
n1 = np.zeros((x, y))
index = 0
for i in normal:
    index2 = 0
    for j in i:
        n1[index, index2] = j[0]
        index2 += 1
    index += 1

n2 = np.zeros((x, y))
index = 0
for i in normal:
    index2 = 0
    for j in i:
        n2[index, index2] = j[1]
        index2 += 1
    index += 1

n3 = np.zeros((x, y))
index = 0
for i in normal:
    index2 = 0
    for j in i:
        n3[index, index2] = j[2]
        index2 += 1
    index += 1

# integrate and display
z = ps.unbiased_integrate(n1, n2, n3, mask)
# z = ps.simchony_integrate(n1, n2, n3, mask)

ps.display_depth(z)

cv2.waitKey(0)
cv2.destroyAllWindows()